import calendar
import glob
import pickle
import re

try:
    import boto3
except:
    print('boto3 library unavailable')
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from torch.utils.data import Dataset


def calc_earnings_markers(trading_dates, earnings_dates):
    """Aligns `trading_dates` with `earnings_dates`
    
    'On'     : Earnings released on that date
    'Before' : Last trading date immediately before earnings released date
    'After'  : First trading date immediately after earnings released date
    'Non'    : Non-earnings date
    'Unk'    : No earnings data
    
    Args:
        trading_dates (ndarray): Sequence of trading dates
        earnings_dates (ndarray): Earnings release dates
        
    Returns:
        earnings_markers (ndarray)
    """
    earnings_markers = np.array(['Unk']*len(trading_dates), dtype=object)
    
    if len(earnings_dates) > 0:
        # Fill earnings data bounds with "Non"
        min_date, max_date = np.min(earnings_dates), np.max(earnings_dates)
        earnings_data_mask = np.logical_and(
            trading_dates >= min_date,
            trading_dates <= max_date
        )
        earnings_markers[earnings_data_mask] = 'Non'
        
        # Fill "Before", "On", "After" markers
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


def get_ret_dist_thresholds(n_targets, tails=[5, 7.5, 10]):
    """Returns target values from normal distribution based
    on an equal spacing of probabilities from 0.5% to 99.5%

    Tails are augmented with "extreme" values 

    Args:
        n_targets (int): Number of threshold targets to generate
        tails (list): Extreme thresholds

    Returns:
        threshold_targets (ndarray): (n_targets,)
    """
    lh_tails = [-t for t in tails[::-1]]
    norms = list(norm.ppf(np.linspace(0.005, 0.995, n_targets - len(tails)*2)))
    rh_tails = tails

    threshold_targets =  np.array(lh_tails + norms + rh_tails)

    return threshold_targets


def transform_prices(df_prices, mean_window=260, std_window=10):
    """Transforms raw price data to model inputs
    
    Args:
        df_prices (DataFrame): Raw price inputs
        window (int): Smoothing window length (in trading days)
        
    Returns:
        df (DataFrame): Transformed inputs
    """
    # Fill any missing data with previous value 
    df_out = df_prices.fillna(method='ffill')
    
    # Calendar markers
    df_out['month'] = df_out.index.month - 1
    dow_mapper = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4,
        }
    df_out['dow'] = np.vectorize(dow_mapper.get)(df_out.index.day_name())
    df_out['trading_day'] = df_out.index.map(get_trading_day)
    df_out['trading_days_left'] = df_out.index.map(get_trading_days_left)

    # Earnings markers
    earnings_marker_mapper = {
        'Unk': 0,
        'Non': 1,
        'Before': 2,
        'On': 3,
        'After': 4
    }
    df_out['earnings_marker'] = np.vectorize(earnings_marker_mapper.get)(df_out['earnings_marker'])
    
    # Previous close & turnover
    prev_close = df_out['Close'].shift(periods=1)
    prev_turnover = df_out['Turnover'].shift(periods=1)
    
    # Log returns
    df_out['CL_return'] = np.log(df_out['Low']/prev_close).fillna(value=0)
    df_out['CC_return'] = np.log(df_out['Close']/prev_close).fillna(value=0)
    df_out['CH_return'] = np.log(df_out['High']/prev_close).fillna(value=0)
    df_out['turnover_return'] = np.log((df_out['Turnover'] + 1)/(prev_turnover + 1)).fillna(value=0)

    # Rolling normalization
    for col in ['CL', 'CC', 'CH', 'turnover']:
        
        # Avg
        df_out[f'{col}_avg'] = df_out[f'{col}_return'].ewm(alpha=1/mean_window, min_periods=mean_window//2).mean()
        
        # Std
        return2 = df_out[f'{col}_return']**2
        df_out[f'{col}_std'] = (return2.ewm(alpha=1/std_window, min_periods=mean_window//2).mean())**0.5
        
        # Norm
        norm = (df_out[f'{col}_return'] - df_out[f'{col}_avg'].shift(periods=1))/df_out[f'{col}_std'].shift(periods=1)
        df_out[f'{col}_norm'] = norm

    df_out['LH_norm'] = df_out['CH_norm'] - df_out['CL_norm']
    
    # Drop NA rows
    df_out = df_out.dropna(how='any', axis='rows')

    # Select columns
    cols = [
        'CL_avg',
        'CL_std',
        'CC_avg',
        'CC_std',
        'CH_avg',
        'CH_std',
        'turnover_avg',
        'turnover_std',
        'month',
        'dow',
        'trading_day',
        'trading_days_left',
        'earnings_marker',
        'CL_norm',
        'CC_norm',
        'CH_norm',
        'LH_norm',
        'turnover_norm',
    ]
    return df_out.loc[:, cols]


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

    Example:
        splits = get_data_splits(
            data_dir='../../../Investing Models/Data/Price Transforms',
            st_date='2018-12-31',
            end_date='2020-12-31',
            window_len=280+65,
            buffer=30,
            p_valid=0.1
        )

        >> splits[0]
        {'buffer_st': Timestamp('2018-12-31 00:00:00', freq='D'),
        'valid_st': Timestamp('2019-01-30 00:00:00', freq='D'),
        'valid_end': Timestamp('2020-01-10 00:00:00', freq='D'),
        'buffer_end': Timestamp('2020-02-09 00:00:00', freq='D'),
        'files_valid': ['../../../Investing Models/Data/Price Transforms/CERN.pkl', ...],
        'files_train': ['../../../Investing Models/Data/Price Transforms/AVGO.pkl', ...]}
    """
    # Start and end of each ticker
    date_spans = []
    for file in glob.glob(f'{data_dir}/*.pkl'):
        df = pd.read_pickle(file)

        if df.index.name == 'Date':
            dt_start, dt_end = min(df.index), max(df.index)
        elif 'Date' in df.columns:
            dt_start, dt_end = min(df['Date']), max(df['Date'])
        else:
            raise ValueError('"Date" not in index or columns')

        date_spans.append([file, dt_start, dt_end])

    date_spans = pd.DataFrame(date_spans)
    date_spans.columns = ['file', 'start', 'end']
    
    # Number of tickers for each validation block
    n_valid = int(len(date_spans)*p_valid)
    
    # Validation dates
    validation_dates = get_validation_dates(st_date, end_date, window_len, buffer)
    
    p = re.compile(r'[^\\\\|\/]{1,100}(?=\.pkl$)')
    splits = []
    for block in validation_dates:
        mask = np.logical_and(
            date_spans['start'] < block['valid_end'],
            date_spans['end'] > block['valid_st']
        )
        files_valid = list(date_spans.loc[mask,'file'].sample(n=n_valid).values)
        files_train = list(set(date_spans['file'].values).difference(files_valid))
        symbols_valid = [p.findall(file)[0] for file in files_valid]
        symbols_train = [p.findall(file)[0] for file in files_train]
        splits.append({
            **block, 
            **{
                'files_valid': files_valid,
                'files_train': files_train,
                'symbols_valid': symbols_valid,
                'symbols_train': symbols_train,
            }
        })
        
    return splits


class FinancialStatementsDataset(Dataset):
    def __init__(
        self, 
        files, 
        date_A, 
        date_B, 
        range_type='exclude', 
        n_historical=20, 
        p_mask=0.2,
        seed=1234,):
        """
        Args:
            files (list): List of paths to transformed files included in the dataset's split
            date_A (str): Either `buffer_st` for training split or `valid_st` for validation split
            date_B (str): Either `buffer_end` for training split or `valid_end` for validation split
            range_type (str): Either 'exclude' for training split or 'include' for validation split
            n_historical (int): Number of financial quarters to use
            p_mask (float): Number entries to mask in input into autoencoder
            seed (int): Random state
        """
        
        self.date_A = date_A
        self.date_B = date_B
        self.range_type = range_type
        self.n_historical = n_historical
        self.p_mask = p_mask
        self.rnd = np.random.RandomState(seed)

        # Statement fields
        self.fields = [
            'AccountsPayable', 
            'AccountsReceivable',
            'AccruedInterestReceivable', 
            'AccumulatedDepreciation',
            'AdditionalPaidInCapital', 
            'AdjustedGeographySegmentData',
            'AllowanceForDoubtfulAccountsReceivable', 
            'AmortizationCashFlow',
            'AmortizationOfIntangibles', 
            'AmortizationOfSecurities',
            'AssetImpairmentCharge', 
            'AssetsHeldForSaleCurrent',
            'AvailableForSaleSecurities', 
            'BeginningCashPosition',
            'BuildingsAndImprovements', 
            'CapitalExpenditure',
            'CapitalExpenditureReported', 
            'CapitalLeaseObligations',
            'CapitalStock', 
            'CashAndCashEquivalents',
            'CashCashEquivalentsAndShortTermInvestments', 
            'CashDividendsPaid',
            'CashEquivalents', 
            'CashFinancial',
            'CashFlowFromContinuingFinancingActivities',
            'CashFlowFromContinuingInvestingActivities',
            'CashFlowFromContinuingOperatingActivities',
            'CashFlowFromDiscontinuedOperation',
            'CashFlowsfromusedinOperatingActivitiesDirect',
            'CashFromDiscontinuedFinancingActivities',
            'CashFromDiscontinuedInvestingActivities',
            'CashFromDiscontinuedOperatingActivities',
            'ChangeInAccountPayable', 
            'ChangeInAccruedExpense',
            'ChangeInDividendPayable', 
            'ChangeInIncomeTaxPayable',
            'ChangeInInterestPayable', 
            'ChangeInInventory',
            'ChangeInOtherCurrentAssets', 
            'ChangeInOtherCurrentLiabilities',
            'ChangeInOtherWorkingCapital', 
            'ChangeInPayable',
            'ChangeInPayablesAndAccruedExpense', 
            'ChangeInPrepaidAssets',
            'ChangeInReceivables', 
            'ChangeInTaxPayable',
            'ChangeInWorkingCapital', 
            'ChangesInAccountReceivables',
            'ChangesInCash', 
            'ClassesofCashPayments',
            'ClassesofCashReceiptsfromOperatingActivities', 
            'CommercialPaper',
            'CommonStock', 
            'CommonStockDividendPaid', 
            'CommonStockEquity',
            'CommonStockIssuance', 
            'CommonStockPayments',
            'ConstructionInProgress', 
            'CurrentAccruedExpenses',
            'CurrentAssets', 
            'CurrentCapitalLeaseObligation', 
            'CurrentDebt',
            'CurrentDebtAndCapitalLeaseObligation', 
            'CurrentDeferredAssets',
            'CurrentDeferredLiabilities', 
            'CurrentDeferredRevenue',
            'CurrentDeferredTaxesAssets', 
            'CurrentDeferredTaxesLiabilities',
            'CurrentLiabilities', 
            'CurrentNotesPayable', 
            'CurrentProvisions',
            'DeferredIncomeTax', 
            'DeferredTax', 
            'DefinedPensionBenefit',
            'Depletion', 
            'Depreciation', 
            'DepreciationAmortizationDepletion',
            'DepreciationAndAmortization', 
            'DerivativeProductLiabilities',
            'DividendPaidCFO', 
            'DividendReceivedCFO', 
            'DividendsPaidDirect',
            'DividendsPayable', 
            'DividendsReceivedCFI',
            'DividendsReceivedDirect', 
            'DomesticSales',
            'DuefromRelatedPartiesCurrent', 
            'DuefromRelatedPartiesNonCurrent',
            'DuetoRelatedPartiesCurrent', 
            'DuetoRelatedPartiesNonCurrent',
            'EarningsLossesFromEquityInvestments',
            'EffectOfExchangeRateChanges', 
            'EmployeeBenefits',
            'EndCashPosition', 
            'ExcessTaxBenefitFromStockBasedCompensation',
            'FinancialAssets',
            'FinancialAssetsDesignatedasFairValueThroughProfitorLossTotal',
            'FinancingCashFlow', 
            'FinishedGoods',
            'FixedAssetsRevaluationReserve',
            'ForeignCurrencyTranslationAdjustments', 
            'ForeignSales',
            'FreeCashFlow', 
            'GainLossOnInvestmentSecurities',
            'GainLossOnSaleOfBusiness', 
            'GainLossOnSaleOfPPE',
            'GainsLossesNotAffectingRetainedEarnings',
            'GeneralPartnershipCapital', 
            'Goodwill',
            'GoodwillAndOtherIntangibleAssets',
            'GrossAccountsReceivable',
            'GrossPPE', 
            'HedgingAssetsCurrent', 
            'HeldToMaturitySecurities',
            'IncomeTaxPaidSupplementalData', 
            'IncomeTaxPayable',
            'InterestPaidCFF', 
            'InterestPaidCFO', 
            'InterestPaidDirect',
            'InterestPaidSupplementalData', 
            'InterestPayable',
            'InterestReceivedCFI', 
            'InterestReceivedCFO',
            'InterestReceivedDirect', 
            'InventoriesAdjustmentsAllowances',
            'Inventory', 
            'InvestedCapital', 
            'InvestingCashFlow',
            'InvestmentProperties', 
            'InvestmentinFinancialAssets',
            'InvestmentsAndAdvances',
            'InvestmentsInOtherVenturesUnderEquityMethod',
            'InvestmentsinAssociatesatCost',
            'InvestmentsinJointVenturesatCost',
            'InvestmentsinSubsidiariesatCost', 
            'IssuanceOfCapitalStock',
            'IssuanceOfDebt', 
            'LandAndImprovements', 
            'Leases',
            'LiabilitiesHeldforSaleNonCurrent', 
            'LimitedPartnershipCapital',
            'LineOfCredit', 
            'LoansReceivable',
            'LongTermCapitalLeaseObligation', 
            'LongTermDebt',
            'LongTermDebtAndCapitalLeaseObligation', 
            'LongTermDebtIssuance',
            'LongTermDebtPayments', 
            'LongTermEquityInvestment',
            'LongTermProvisions', 
            'MachineryFurnitureEquipment',
            'MinimumPensionLiabilities', 
            'MinorityInterest',
            'NetBusinessPurchaseAndSale', 
            'NetCommonStockIssuance', 
            'NetDebt',
            'NetForeignCurrencyExchangeGainLoss',
            'NetIncomeFromContinuingOperations',
            'NetIntangiblesPurchaseAndSale',
            'NetInvestmentPropertiesPurchaseAndSale',
            'NetInvestmentPurchaseAndSale', 
            'NetIssuancePaymentsOfDebt',
            'NetLongTermDebtIssuance', 
            'NetOtherFinancingCharges',
            'NetOtherInvestingChanges', 
            'NetPPE', 
            'NetPPEPurchaseAndSale',
            'NetPreferredStockIssuance', 
            'NetShortTermDebtIssuance',
            'NetTangibleAssets', 
            'NonCurrentAccountsReceivable',
            'NonCurrentAccruedExpenses', 
            'NonCurrentDeferredAssets',
            'NonCurrentDeferredLiabilities', 
            'NonCurrentDeferredRevenue',
            'NonCurrentDeferredTaxesAssets',
            'NonCurrentDeferredTaxesLiabilities', 
            'NonCurrentNoteReceivables',
            'NonCurrentPensionAndOtherPostretirementBenefitPlans',
            'NonCurrentPrepaidAssets', 
            'NotesReceivable', 
            'OperatingCashFlow',
            'OperatingGainsLosses', 
            'OrdinarySharesNumber',
            'OtherCapitalStock', 
            'OtherCashAdjustmentInsideChangeinCash',
            'OtherCashAdjustmentOutsideChangeinCash',
            'OtherCashPaymentsfromOperatingActivities',
            'OtherCashReceiptsfromOperatingActivities', 
            'OtherCurrentAssets',
            'OtherCurrentBorrowings', 
            'OtherCurrentLiabilities',
            'OtherEquityAdjustments', 
            'OtherEquityInterest',
            'OtherIntangibleAssets', 
            'OtherInventories', 
            'OtherInvestments',
            'OtherNonCashItems', 
            'OtherNonCurrentAssets',
            'OtherNonCurrentLiabilities', 
            'OtherPayable', 
            'OtherProperties',
            'OtherReceivables', 
            'OtherShortTermInvestments', 
            'Payables',
            'PayablesAndAccruedExpenses', 
            'PaymentsonBehalfofEmployees',
            'PaymentstoSuppliersforGoodsandServices',
            'PensionAndEmployeeBenefitExpense',
            'PensionandOtherPostRetirementBenefitPlansCurrent',
            'PreferredSecuritiesOutsideStockEquity', 
            'PreferredSharesNumber',
            'PreferredStock', 
            'PreferredStockDividendPaid',
            'PreferredStockEquity', 
            'PreferredStockIssuance',
            'PreferredStockPayments', 
            'PrepaidAssets',
            'ProceedsFromStockOptionExercised', 
            'Properties',
            'ProvisionandWriteOffofAssets', 
            'PurchaseOfBusiness',
            'PurchaseOfIntangibles', 
            'PurchaseOfInvestment',
            'PurchaseOfInvestmentProperties', 
            'PurchaseOfPPE', 
            'RawMaterials',
            'ReceiptsfromCustomers', 
            'ReceiptsfromGovernmentGrants',
            'Receivables', 
            'ReceivablesAdjustmentsAllowances',
            'RepaymentOfDebt', 
            'RepurchaseOfCapitalStock', 
            'RestrictedCash',
            'RestrictedCommonStock', 
            'RetainedEarnings', 
            'SaleOfBusiness',
            'SaleOfIntangibles', 
            'SaleOfInvestment',
            'SaleOfInvestmentProperties', 
            'SaleOfPPE', 
            'ShareIssued',
            'ShortTermDebtIssuance', 
            'ShortTermDebtPayments',
            'StockBasedCompensation', 
            'StockholdersEquity',
            'TangibleBookValue', 
            'TaxesReceivable', 
            'TaxesRefundPaid',
            'TaxesRefundPaidDirect', 
            'TotalAssets', 
            'TotalCapitalization',
            'TotalDebt', 
            'TotalEquityGrossMinorityInterest',
            'TotalLiabilitiesNetMinorityInterest', 
            'TotalNonCurrentAssets',
            'TotalNonCurrentLiabilitiesNetMinorityInterest',
            'TotalPartnershipCapital', 
            'TotalTaxPayable',
            'TradeandOtherPayablesNonCurrent', 
            'TradingSecurities',
            'TreasurySharesNumber', 
            'TreasuryStock', 
            'UnrealizedGainLoss',
            'UnrealizedGainLossOnInvestmentSecurities', 
            'WorkInProcess',
            'WorkingCapital'
        ]

        # Enumerate all items (tickers x # rolling window for each ticker)
        self.items = []
        for file in files:
            df = pd.read_pickle(file)
            self.items.extend([
                (file, st_idx, end_idx)
                for st_idx, end_idx
                in self._get_rolling_window_idxs(df)
                if len(df.index) >= self.n_historical + 1
            ])
                
    def __len__(self):
        return len(self.items)
                                          
    def __getitem__(self, idx):
        "Load transformed DataFrame, slice to window and extact tensors"
        file, st_idx, end_idx = self.items[idx]
        df = pd.read_pickle(file)
        return self._get_inputs(df.iloc[st_idx:end_idx].copy(deep=False))
    
    def _get_rolling_window_idxs(self, df):
        "Get index tuple [st, end) of rolling windows for a seq of dates"
        dates = df.index
        windows = []
        for idx, dt in enumerate(dates[self.n_historical:], self.n_historical):
            if ((self.range_type == 'exclude' and (dt < self.date_A or dt > self.date_B)) or
                (self.range_type == 'include' and dt >= self.date_A and dt <= self.date_B)):
                    df_window = df.iloc[idx - self.n_historical:idx]
                    if np.all(df_window['TotalAssets'].values > 0): # Total assets must be > 0
                        x = df_window.values
                        if np.all(np.isfinite(x)):    # All values must be finite
                            if np.mean(x == 0) < 0.9: # Must have >= 10% non-zero entries
                                windows.append((idx - self.n_historical, idx))

        return windows
        
    def _get_inputs(self, df):
        "Partition transformed DataFrame into input and target tensors"

        df = df.loc[:, self.fields]

        total_assets_med = df['TotalAssets'].median()
        targets = np.minimum(np.maximum(-2, (df/total_assets_med).values), 2)

        mask = self.rnd.rand(*targets.shape) <= self.p_mask

        inputs = targets*(1 - mask)

        return {
            'inputs': torch.tensor(inputs, dtype=torch.float),
            'targets': torch.tensor(targets, dtype=torch.float)
        }


class PriceSeriesDataset(Dataset):
    """Price Series Dataset

    Examples:
        dataset = PriceSeriesDataset(
            symbols=split['symbols_train'],
            files=split['files_train'],
            date_A=split['buffer_st'],
            date_B=split['buffer_end'],
            range_type='exclude', 
        )

        >> dataset[0]
        {'historical_seq_emb': tensor([[8,  3, 20,  1, 0], ...]]),
         'historical_seq_LCH': tensor([[ -1.23,  0.42,  -0.123], ...]]),
         'historical_seq_LCH_masked': tensor([[ 0.0000,  0.0000,  0.0000], ...]]),
         'recon_loss_mask': tensor([[1, 0, 1, 1, ..]]),
         'future_seq_emb': tensor([[10,  3,  5, 16, 1], ...]]),
         'future_seq_LCH': tensor([[ 0.9524,  0.7966,  1.0035], ...]),
         'future_ret_dists': tensor([[0., 0., ...]])}

        >> dataset.__getitem__(0, 'AMZN')


        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        batch = next(iter(loader))

        for batch in loader:
             ...
    """
    def __init__(
        self,
        symbols,
        files,
        s3_bucket_name=None,
        range_type='none',  
        date_A=None, 
        date_B=None, 
        n_historical=4*256, 
        n_future=65, 
        n_dist_targets=13,
        p_mask=0.2,
        seed=1234):
        """
        Args:
            symbols (list): List of symbols included in the dataset's split
            files (list): List of paths to transformed files included in the dataset's split
            s3_bucket_name (str): Name of AWS s3 bucket, if loading data from there. Default to `None`
            range_type (str): Default to 'none` for no filtering by dates.
                              Or either 'exclude' for training split or 'include' for validation split
            date_A (str): Default to None. If `range_type` is not 'none' then:
                          Either `buffer_st` for training split or `valid_st` for validation split
            date_B (str): Default to None. If `range_type` is not 'none' then:
                          Either `buffer_end` for training split or `valid_end` for validation split

            n_historical (int): Number of trading days to use for historical context
            n_future (int): Number of trading days to predict into the future
            n_dist_targets (int): Number of CDF slices to predict for each future time step
            p_mask (float): Percentage of trading days to mask
            mask_auto_corr (float): 1-period autocorrelation between masking probabilities
            seed (int): Random state
        """
        self.bucket_name = s3_bucket_name
        if s3_bucket_name:
             # AWS s3 client
            self.s3_client = boto3.client('s3')
            
        self.date_A = date_A
        self.date_B = date_B
        self.range_type = range_type
        self.n_historical = n_historical
        self.n_future = n_future
        self.window_len = n_historical + n_future
        self.n_dist_targets = n_dist_targets
        self.p_mask = p_mask
        self.rnd = np.random.RandomState(seed)
        
        self.target_thresholds = get_ret_dist_thresholds(n_dist_targets)
        
        self.cols_emb = [
            'month',
            'dow',
            'trading_day',
            'trading_days_left',
            'earnings_marker'
        ]
        self.cols_LCH = [
            'CL_norm',
            'CC_norm',
            'CH_norm',
        ]
        
        # Enumerate all windows (tickers x # rolling window for each ticker)
        self.symbol_windows, self.all_windows = dict(), []
        for symbol, file in zip(symbols, files):
            df = self._read_file(file)
            windows = [
                (file, st_idx, end_idx)
                for st_idx, end_idx
                in self._get_rolling_window_idxs(df)
            ]
            self.symbol_windows[symbol] = windows
            self.all_windows.extend(windows)
                
    def __len__(self):
        return len(self.all_windows)
                                          
    def __getitem__(self, idx, symbol=None):
        "Load transformed DataFrame, slice to window and extact tensors"
        if symbol:
            file, st_idx, end_idx = self.symbol_windows[symbol][idx]
        else:
            file, st_idx, end_idx = self.all_windows[idx]
        
        df = self._read_file(file)

        df['trading_day'] = np.minimum(22, df['trading_day'])
        df['trading_days_left'] = np.minimum(22, df['trading_days_left'])
        
        return self._get_inputs(df.iloc[st_idx:end_idx].copy(deep=False))
    
    def _read_file(self, file):
        if self.bucket_name:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file)
            if response:
                body = response['Body']
                return pickle.loads(body.read())
        else:
            return pd.read_pickle(file)

    def _get_rolling_window_idxs(self, df):
        "Get index tuple [st, end) of rolling windows for a seq of dates"
        dates = df.index
        windows = []
        for idx in range(self.window_len + self.rnd.randint(0, self.n_future), len(dates), self.n_future):
            if self.range_type == 'none':
                windows.append((idx - self.window_len, idx))

            elif ((self.range_type == 'exclude' and (dates[idx] < self.date_A or dates[idx] > self.date_B)) or
                  (self.range_type == 'include' and dates[idx] >= self.date_A and dates[idx] <= self.date_B)):
                windows.append((idx - self.window_len, idx))

        return windows
        
    def _get_inputs(self, df):
        "Partition transformed DataFrame into input and target tensors"
        
        # Historical seq embedding columns
        historical_seq_emb = torch.tensor(df.iloc[:self.n_historical][self.cols_emb].values, dtype=torch.long)

        # Historical sequence low, close, high normalized returns
        historical_seq_LCH = df.iloc[:self.n_historical][self.cols_LCH].values
        
        # Masked historical sequence low, close, high normalized returns
        rand_seq = self.rnd.rand(self.n_historical)
        mask = rand_seq <= self.p_mask
        historical_seq_LCH_masked = historical_seq_LCH.copy()
        historical_seq_LCH_masked[mask] = 0
        
        historical_seq_LCH = torch.tensor(historical_seq_LCH, dtype=torch.float)
        historical_seq_LCH_masked = torch.tensor(historical_seq_LCH_masked, dtype=torch.float)

        # Reconstruction loss mask - includes all masked time periods plus an additional 25%*p_mask of non masked periods
        recon_loss_mask = torch.tensor(rand_seq <= 1.25*self.p_mask, dtype=torch.float).unsqueeze(-1)
        
        # Future sequence embedding columns
        future_seq_emb = torch.tensor(df.iloc[self.n_historical:][self.cols_emb].values, dtype=torch.long)
        
        # Future sequence low, close, high normalized returns
        future_seq_LCH = torch.from_numpy(
            df.iloc[self.n_historical:][self.cols_LCH].values
        ).to(torch.float)

        # Future cumulative return distributions
        cumulative_CC_norms = np.cumsum(df.iloc[self.n_historical:][self.cols_LCH[1]].values)
        cumulative_CC_norms /= np.sqrt(np.arange(self.n_future) + 1)
        future_ret_dists = torch.from_numpy(np.stack([
            (cumulative_CC_norms <= t).astype(np.float32)
            for t 
            in self.target_thresholds
        ], axis=1))

        return {
            'historical_seq_emb': historical_seq_emb,
            'historical_seq_LCH': historical_seq_LCH,
            'historical_seq_LCH_masked': historical_seq_LCH_masked,
            'recon_loss_mask': recon_loss_mask,
            'future_seq_emb': future_seq_emb,
            'future_seq_LCH': future_seq_LCH,
            'future_ret_dists': future_ret_dists,
        }
