import calendar
import glob

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from torch.utils.data import Dataset


def calc_earnings_markers(trading_dates, earnings_dates):
    """Aligns `trading_dates` with `earnings_dates`
    
    'On'    : Earnings released on that date
    'Before': Last trading date immediately before earnings released date
    'After'  : First trading date immediately after earnings released date
    'Other' : Non-earnings date
    
    Args:
        trading_dates (ndarray): Sequence of trading dates
        earnings_dates (ndarray): Earnings release dates
        
    Returns:
        earnings_markers (ndarray)
    """
    earnings_markers = np.array(['Other']*len(trading_dates), dtype=np.object)
    for e_dt in earnings_dates:
        # Number of days between earnings release date and all trading days
        n_days = (trading_dates - e_dt)/np.timedelta64(1, 'D')
        
        # Dates before release date
        before_idx = np.argmin(np.where(n_days < 0, np.abs(n_days), np.inf))
        earnings_markers[before_idx] = 'Before'
        
        # Date on release date
        earnings_markers[n_days == 0] = 'On'
        
        # Dates after release date
        after_idx = np.argmin(np.where(n_days > 0, n_days, np.inf))
        earnings_markers[after_idx] = 'After'
    
    return earnings_markers


def rolling_window(a, window):
    "NumPy rollling window"
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _get_trading_days(date):
    "Enumerate weekdays for the `date` month"
    trading_dates = np.array([
        dt
        for dt
        in pd.date_range(
            start=f'{date.year}-{date.month}-01',
            end=f'{date.year}-{date.month}-{calendar.monthlen(date.year, date.month)}',
            freq='1D')
        if dt.day_of_week <= 4
    ])
    return trading_dates


def get_trading_day(dt):
    "Trading day number for date"
    trading_dates = _get_trading_days(dt)
    return np.argwhere(trading_dates == dt).squeeze().item()


def get_trading_days_left(dt):
    "Trading days left in month after date"
    trading_dates = _get_trading_days(dt)
    return np.sum(trading_dates > dt)


def cdf_regression(returns, start=0.1, stop=3, num=25, deg=2):
    """Computes the coefficients of least squares polynomial fit
    for the density of the normal distribution vs the observed
    density of the return distribution at a sequence of thresholds.
    
    Args:
        returns (ndarray)
        start (float, optional): Starting threshold
        stop (float, optional): Ending threshold
        num (int, optional): Number of thresholds
        deg (int, optional): Degree of the fitting polynomial
        
    Returns:
        coefs (ndarray)
    """
    thresholds = np.linspace(start, stop, num)
    
    x = norm.cdf(thresholds) - norm.cdf(-thresholds)
    
    std = np.mean(returns**2)**0.5
    abs_rets = np.abs(returns)
    y = [np.mean(abs_rets <= std*t) for t in thresholds]
    
    return np.polyfit(x, y, deg)


def transform(df_prices, df_rates, earnings_dates, mean_window=260, std_window=65):
    """Transforms raw price data to model inputs
    
    Args:
        df_prices (DataFrame): Raw price inputs
        df_rates (DataFrame)
        earnings_dates (ndarray): Earnings release dates
        window (int): Smoothing window length (in trading days)
        
    Returns:
        df (DataFrame): Transformed inputs
    """
    # Fill any missing data with previous value 
    df_prices = df_prices.fillna(method='ffill')

    # Convert Date to datetime
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    
    # Add rates
    df = pd.merge_ordered(df_prices, df_rates, on='Date', how='left')
    
    # Get earnings markers
    df['earnings_marker'] = calc_earnings_markers(df['Date'].values, earnings_dates)
    
    # Calendar markers
    df['month'] = df['Date'].dt.month
    df['dow'] = df['Date'].dt.day_name()
    df['trading_day'] = df['Date'].apply(get_trading_day)
    df['trading_days_left'] = df['Date'].apply(get_trading_days_left)
    
    # Adjust high and low data
    df['adj_factor'] = df['Adj Close']/df['Close']
    df['adj_high'] = np.maximum(df['Adj Close'], df['adj_factor']*df['High'])
    df['adj_low'] = np.minimum(df['Adj Close'], df['adj_factor']*df['Low'])
    
    # Shift close and risk free rate
    prev_close = df['Adj Close'].shift(periods=1)
    prev_rfr = df['risk_free_rate'].shift(periods=1)/252
    
    # Excess log returns
    df['CL_return'] = np.log(df['adj_low']/prev_close).fillna(value=0) - prev_rfr
    df['CC_return'] = np.log(df['Adj Close']/prev_close).fillna(value=0) - prev_rfr
    df['CH_return'] = np.log(df['adj_high']/prev_close).fillna(value=0) - prev_rfr

    # Rolling normalization
    for col in ['CL', 'CC', 'CH']:
        
        # Avg
        df[f'{col}_avg'] = df[f'{col}_return'].ewm(alpha=1/mean_window, min_periods=mean_window//2).mean()
        
        # Std
        return2 = df[f'{col}_return']**2
        df[f'{col}_std'] = (return2.ewm(alpha=1/std_window, min_periods=mean_window//2).mean())**0.5
        
        # Norm
        norm = (df[f'{col}_return'] - df[f'{col}_avg'].shift(periods=1))/df[f'{col}_std'].shift(periods=1)
        df[f'{col}_norm'] = norm

    df['LH_norm'] = df['CH_norm'] - df['CL_norm']
    
    # Drop NA rows
    df = df.dropna(how='any', axis='rows')

    # Select columns
    cols = [
        'Date',
        'Symbol',
        'risk_free_rate',
        'CL_avg',
        'CL_std',
        'CC_avg',
        'CC_std',
        'CH_avg',
        'CH_std',
        'month',
        'dow',
        'trading_day',
        'trading_days_left',
        'earnings_marker',
        'CL_norm',
        'CC_norm',
        'CH_norm',
        'LH_norm',
    ]
    return  df.loc[:, cols]


def get_validation_dates(st_date, end_date, window_len, buffer):
    """Sequential, non-overlapping blocks of validation dates

    Args:
        st_date (str): First date in the first validation block
        end_date (str): Last date in the last validation block
        window_len (int): Number of calendar dates in each block
        buffer (int): Number of calendar dates to exclude adjacent (before & after) each block for training data

    Returns
        validation_dates (list): Sequence of validation blocks
    """
    dates = pd.date_range(st_date, end_date, freq='D')
    
    validation_dates = [
        {
            'buffer_st': dates[i - buffer],
            'valid_st': dates[i],
            'valid_end': dates[i + window_len],
            'buffer_end': dates[i + window_len + buffer]
        }
        for i 
        in range(buffer, len(dates) - window_len - buffer, window_len)
    ]
    return validation_dates


def get_data_splits(data_dir, st_date, end_date, window_len, buffer, p_valid):
    """Split data files into blocks of training and validation data.

    Purge files in validation split from training split.

    Args:
        data_dir (str): Path where transformed PKL DataFrames are saved
        st_date (str): First date in the first validation block
        end_date (str): Last date in the last validation block
        window_len (int): Number of calendar dates in each block
        buffer (int): Number of calendar dates to exclude adjacent (before & after) each block for training data
        p_valid (float): Percentage of data files to use for validation data

    Returns:
        splits (list): Sequence of data splits
    """
    # Start and end of each ticker
    date_spans = []
    for file in glob.glob(f'{data_dir}/*.pkl'):
        df = pd.read_pickle(file)
        dt_start, dt_end = min(df['Date']), max(df['Date'])
        date_spans.append([file, dt_start, dt_end])

    date_spans = pd.DataFrame(date_spans)
    date_spans.columns = ['file', 'start', 'end']
    
    # Number of tickers for each validation block
    n_valid = int(len(date_spans)*p_valid)
    
    # Validation dates
    validation_dates = get_validation_dates(st_date, end_date, window_len, buffer)
    
    splits = []
    for block in validation_dates:
        mask = np.logical_and(
            date_spans['start'] < block['valid_end'],
            date_spans['end'] > block['valid_st']
        )
        files_valid = list(date_spans.loc[mask,'file'].sample(n=n_valid).values)
        files_train = list(set(date_spans['file'].values).difference(files_valid))
        splits.append({**block, **{'files_valid': files_valid,'files_train': files_train}})
        
    return splits


class StockSeriesDataset(Dataset):
    def __init__(
        self, 
        files, 
        date_A, 
        date_B, 
        range_type='exclude', 
        n_historical=5*250, 
        n_future=65, 
        n_targets=51):
        """
        Args:
            files (list): List of paths to transformed files included in the dataset's split
            date_A (str): Either `buffer_st` for training split or `valid_st` for validation split
            date_B (str): Either `buffer_end` for training split or `valid_end` for validation split
            range_type (str): Either 'exclude' for training split or 'include' for validation split
            n_historical (int): Number of trading days to use for historical context
            n_future (int): Number of trading days to predict into the future
            n_targets (int): Number of CDF slices to predict for each future time step
        """
        
        self.date_A = date_A
        self.date_B = date_B
        self.range_type = range_type
        self.n_historical = n_historical
        self.n_future = n_future
        self.window_len = n_historical + n_future
        self.n_targets = n_targets
        self.target_thresholds = norm.ppf(np.linspace(0.001, 0.999, n_targets))
        
        self.dow_mapper = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4,
        }
        self.earnings_marker_mapper = {
            'Other': 0,
            'Before': 1,
            'On': 2,
            'After': 3
        }
        
        self.cols_norm = [
            'risk_free_rate', 
            'return_log_avg_120d', 
            'tr_avg_20d'
        ]
        self.cols_emb = [
            'month',
            'dow',
            'trading_day',
            'trading_days_left',
            'earnings_marker',
        ]
        self.cols_float = [
            'return_norm',
            'high_to_close_norm',
            'low_to_close_norm',
            'true_range_norm',
            'tr_avg_norm',
            'choppiness_260d',
            'cdf_x2',
            'cdf_x',
            'cdf_I',
        ]
        
        # Enumerate all items (tickers x # rolling window for each ticker)
        self.items = []
        for file in files:
            df = pd.read_pickle(file)
            self.items.extend([
                (file, st_idx, end_idx)
                for st_idx, end_idx
                in self._get_rolling_window_idxs(df['Date'])
            ])
                
    def __len__(self):
        return len(self.items)
                                          
    def __getitem__(self, idx):
        "Load transformed DataFrame, slice to window and extact tensors"
        file, st_idx, end_idx = self.items[idx]
        df = pd.read_pickle(file)
        return self._get_inputs(df.iloc[st_idx:end_idx].copy(deep=False))
    
    def _get_rolling_window_idxs(self, dates):
        "Get index tuple [st, end) of rolling windows for a seq of dates"

        windows = []
        for idx, dt in enumerate(dates[self.window_len:], self.window_len):
            if ((self.range_type == 'exclude' and (dt < self.date_A or dt > self.date_B)) or
                (self.range_type == 'include' and dt >= self.date_A and dt <= self.date_B)):
                    windows.append((idx - self.window_len, idx))

        return windows
        
    def _get_inputs(self, df):
        "Partition transformed DataFrame into input and target tensors"

        df['dow'] = np.vectorize(self.dow_mapper.get)(df['dow'])
        df['earnings_marker'] = np.vectorize(self.earnings_marker_mapper.get)(df['earnings_marker'])
        
        historical_seq_emb = torch.tensor(df.iloc[:self.n_historical][self.cols_emb].values, dtype=torch.long)
        historical_seq_float = torch.tensor(df.iloc[:self.n_historical][self.cols_float].values, dtype=torch.float)
        future_seq_emb = torch.tensor(df.iloc[self.n_historical:][self.cols_emb].values, dtype=torch.long)

        rfr, mean, std = df.iloc[self.n_historical - 1][self.cols_norm].values
        daily_rfr = np.exp(rfr/252) - 1
        future_rets = np.log(df.iloc[self.n_historical:]['return'].values + 1)
        excess_rets = future_rets - daily_rfr - mean
        cumulative_rets = np.exp(np.cumsum(excess_rets)) - 1
        normalized_rets = cumulative_rets/(std*np.sqrt(np.arange(self.n_future) + 1))
        targets = torch.from_numpy(np.stack([
            (normalized_rets <= t).astype(np.float32)
            for t 
            in self.target_thresholds
        ], axis=1))

        return {
            'historical_seq_emb': historical_seq_emb,
            'historical_seq_float': historical_seq_float,
            'future_seq_emb': future_seq_emb,
            'targets': targets
        }