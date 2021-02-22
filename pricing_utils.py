import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def adj_spot_for_dvd(spot, dvd_amts, risk_free_rates, cdays_exdvds, days_in_yr=365):
    """Adjust spot price for discrete dividends
    
    Args:
        spot (float)                : Spot price
        dvd_amts (list)             : Dividend amounts
        risk_free_rates (list)      : Annual risk free rate for the period until each ex-dividend dates
        cdays_exdvds (list)         : Total number of calendar days until ex-dividend for each dividend
        days_in_yr (float, optional): Total number of calendar days in the year (normal years: 365 days, leap years: 366)
        
    Returns:
        spot_dvd_adj (float)
    """
    assert len(dvd_amts) == len(risk_free_rates), 'Number of dividend payments needs to match number of rates' 
    assert len(dvd_amts) == len(cdays_exdvds), 'Number of dividend payments needs to match ex days' 
    
    dvd_amt_pv = 0
    for dvd_amt, risk_free_rate, cdays_exdvd in zip(dvd_amts, risk_free_rates, cdays_exdvds):
        t_exdvd = cdays_exdvd/days_in_yr                       # Fraction of year until ex-dividend
        dvd_amt_pv += dvd_amt*np.exp(-risk_free_rate*t_exdvd)  # Present value of dividend
    
    spot_dvd_adj = spot - dvd_amt_pv  # Remove pv of all dividends from current price
    
    return spot_dvd_adj


def sample_bsm_future_return_distribution(
    n, 
    volatility, 
    risk_free_rate_optexp,
    cdays_optexp,
    tdays_optexp=None,
    days_in_yr=365, 
    trading_days_in_yr=252,
    seed=1234):
    """Sample from BSM arbitrage free future return distribution

    Args:
        n (int)                             : Number of returns to sample
        volatility (float)                  : Annualized volatility
        risk_free_rate_optexp (float)       : Annual risk free rate for the period until option expiration
        cdays_optexp (float)                : Total number of calendar days until option expiration
        tdays_optexp (float, optional)      : Total number of trading days until option expiration. 
                                              Used when volatility is estimated from trading activity
        days_in_yr (float, optional)        : Total number of calendar days in the year (normal years: 365 days, leap years: 366)
        trading_days_in_yr (float, optional): Total number of trading days in the year
        seed (int, optional)                : Random state

    Returns:
        returns (ndarray)

    """
    
    rnd = np.random.RandomState(seed)
    normals = rnd.randn(n)
    
    # Fraction of year until option expiration
    t_optexp_calendar = cdays_optexp/days_in_yr
    if tdays_optexp:
        t_optexp_trading = tdays_optexp/trading_days_in_yr
    else:
        t_optexp_trading = t_optexp_calendar
    
    returns = risk_free_rate_optexp*t_optexp_calendar
    returns -= volatility**2/2*t_optexp_trading
    returns -= volatility*normals*t_optexp_trading**0.5
    returns = np.exp(returns) - 1
    
    return returns


def sample_from_bins(n, bins, bin_ppfs):
    """Random sample using linear interpolation between bins based on bin ppfs
    
    Args:
        n (int): Number of samples
        bins (ndarray): Bin thresholds (<=)
        bin_ppfs (ndarray): Emperical or predicted ppf for each bin
    """
    
    # Sample uniform random numbers
    x = np.random.rand(n)
    
    # Find upper and lower bin indexes for each random draw based on ppfs
    u_idx = np.searchsorted(bin_ppfs, x, side='left', sorter=None)
    l_idx = u_idx - 1
    
    # PPFs
    u_ppf = bin_ppfs[u_idx]
    l_ppf = bin_ppfs[l_idx]
    
    # Values based on linear interpolation between upper and lower bins
    u_bin = bins[u_idx]
    l_bin = bins[l_idx]
    interp_pct = (x - l_ppf)/(u_ppf - l_ppf)
    values = (u_bin - l_bin)*interp_pct + l_bin
    
    return values


def price_call_bsm(
    spot, 
    strike, 
    volatility,
    risk_free_rate_optexp,
    cdays_optexp,
    tdays_optexp=None,
    dvd_amts=[], 
    risk_free_rates_exdvd=[],
    cdays_exdvds=[], 
    days_in_yr=365, 
    trading_days_in_yr=252):
    """BSM call price with discrete dividends
    
    Args:
        spot (float)                        : Spot price
        strike (float)                      : Strike price
        volatility (float)                  : Annualized volatility
        risk_free_rate_optexp (float)       : Annual risk free rate for the period until option expiration
        cdays_optexp (float)                : Total number of calendar days until option expiration
        tdays_optexp (float, optional)      : Total number of trading days until option expiration. 
                                              Used when volatility is estimated from trading activity
        dvd_amts (list)                     : Dividend amounts
        risk_free_rates_exdvd (list)        : Annual risk free rate for the period until each ex-dividend date
        cdays_exdvds (list)                 : Total number of calendar days until ex-dividend for each dividend
        days_in_yr (float, optional)        : Total number of calendar days in the year (normal years: 365 days, leap years: 366)
        trading_days_in_yr (float, optional): Total number of trading days in the year
        
    Returns:
        price (float), delta (float)
    """
    
    # Fraction of year until option expiration
    t_optexp_calendar = cdays_optexp/days_in_yr
    if tdays_optexp:
        t_optexp_trading = tdays_optexp/trading_days_in_yr
    else:
        t_optexp_trading = t_optexp_calendar
    
    # Adjust spot for discrete dividend
    spot_dvd_adj = adj_spot_for_dvd(spot, dvd_amts, risk_free_rates_exdvd, cdays_exdvds)
    
    # d1 and d2
    d1 = np.log(spot_dvd_adj/strike) 
    d1 += risk_free_rate_optexp*t_optexp_calendar 
    d1 += volatility**2/2*t_optexp_trading
    d1 /= (volatility*t_optexp_trading**0.5)
    d2 = d1 - volatility*t_optexp_trading**0.5
    
    price = spot_dvd_adj*norm.cdf(d1)
    price -= strike*np.exp(-risk_free_rate_optexp*t_optexp_calendar)*norm.cdf(d2)
    
    return price, norm.cdf(d1)


def price_put_bsm(
    spot, 
    strike, 
    volatility,
    risk_free_rate_optexp,
    cdays_optexp,
    tdays_optexp=None,
    dvd_amts=[], 
    risk_free_rates_exdvd=[],
    cdays_exdvds=[], 
    days_in_yr=365, 
    trading_days_in_yr=252):
    """BSM put price with discrete dividends
    
    Args:
        spot (float)                        : Spot price
        strike (float)                      : Strike price
        volatility (float)                  : Annualized volatility
        risk_free_rate_optexp (float)       : Annual risk free rate for the period until option expiration
        cdays_optexp (float)                : Total number of calendar days until option expiration
        tdays_optexp (float, optional)      : Total number of trading days until option expiration. 
                                              Used when volatility is estimated from trading activity
        dvd_amts (list)                     : Dividend amounts
        risk_free_rates_exdvd (list)        : Annual risk free rate for the period until each ex-dividend date
        cdays_exdvds (list)                 : Total number of calendar days until ex-dividend for each dividend
        days_in_yr (float, optional)        : Total number of calendar days in the year (normal years: 365 days, leap years: 366)
        trading_days_in_yr (float, optional): Total number of trading days in the year
        
    Returns:
        price (float), delta (float)
    """
    
    # Fraction of year until option expiration
    t_optexp_calendar = cdays_optexp/days_in_yr
    if tdays_optexp:
        t_optexp_trading = tdays_optexp/trading_days_in_yr
    else:
        t_optexp_trading = t_optexp_calendar
    
    # Adjust spot for discrete dividend
    spot_dvd_adj = adj_spot_for_dvd(spot, dvd_amts, risk_free_rates_exdvd, cdays_exdvds)
    
    # d1 and d2
    d1 = np.log(spot_dvd_adj/strike) 
    d1 += risk_free_rate_optexp*t_optexp_calendar 
    d1 += volatility**2/2*t_optexp_trading
    d1 /= (volatility*t_optexp_trading**0.5)
    d2 = d1 - volatility*t_optexp_trading**0.5
    
    price = strike*np.exp(-risk_free_rate_optexp*t_optexp_calendar)*norm.cdf(-d2)
    price -= spot_dvd_adj*norm.cdf(-d1)
    
    return price, norm.cdf(d1) - 1


def price_call_pcp(
    put, 
    spot, 
    strike,
    risk_free_rate_optexp,
    cdays_optexp,
    dvd_amts=[], 
    risk_free_rates_exdvd=[], 
    cdays_exdvds=[], 
    days_in_yr=365):
    """Call price from put-call parity
    
     Args:
        put (float)                         : Price of put option with same strike & expiration
        spot (float)                        : Spot price
        strike (float)                      : Strike price
        volatility (float)                  : Annualized volatility
        risk_free_rate_optexp (float)       : Annual risk free rate for the period until option expiration
        cdays_optexp (float)                : Total number of calendar days until option expiration
        dvd_amts (list)                     : Dividend amounts
        risk_free_rates_exdvd (list)        : Annual risk free rate for the period until each ex-dividend date
        cdays_exdvds (list)                 : Total number of calendar days until ex-dividend for each dividend
        days_in_yr (float, optional)        : Total number of calendar days in the year (normal years: 365 days, leap years: 366)
        
    Returns:
        price (float)
    """
    spot_dvd_adj = adj_spot_for_dvd(spot, dvd_amts, risk_free_rates_exdvd, cdays_exdvds)
    
    t_optexp = cdays_optexp/days_in_yr
    strike_pv = strike*np.exp(-risk_free_rate_optexp*t_optexp)
    
    price = put + spot_dvd_adj - strike_pv
    
    return price


def price_put_pcp(
    call, 
    spot, 
    strike,
    risk_free_rate_optexp,
    cdays_optexp,
    dvd_amts=[], 
    risk_free_rates_exdvd=[], 
    cdays_exdvds=[], 
    days_in_yr=365):
    """Call price from put-call parity
    
     Args:
        call (float)                        : Price of call option with same strike & expiration
        spot (float)                        : Spot price
        strike (float)                      : Strike price
        volatility (float)                  : Annualized volatility
        risk_free_rate_optexp (float)       : Annual risk free rate for the period until option expiration
        cdays_optexp (float)                : Total number of calendar days until option expiration
        dvd_amts (list)                     : Dividend amounts
        risk_free_rates_exdvd (list)        : Annual risk free rate for the period until each ex-dividend date
        cdays_exdvds (list)                 : Total number of calendar days until ex-dividend for each dividend
        days_in_yr (float, optional)        : Total number of calendar days in the year (normal years: 365 days, leap years: 366)
        
    Returns:
        price (float)
    """
    spot_dvd_adj = adj_spot_for_dvd(spot, dvd_amts, risk_free_rates_exdvd, cdays_exdvds)
    
    t_optexp = cdays_optexp/days_in_yr
    strike_pv = strike*np.exp(-risk_free_rate_optexp*t_optexp)
    
    price = call - spot_dvd_adj + strike_pv
    
    return price


def price_callput_replicating(
    future_return_distribution,
    spot, 
    strike, 
    risk_free_rate_optexp,
    cdays_optexp,
    dvd_amts=[], 
    risk_free_rates_exdvd=[],
    cdays_exdvds=[], 
    days_in_yr=365):
    """Call and put prices using replicating portfolios based on a 
    forecasted return distribution at option expiration.

    Args:
        future_return_distribution (ndarray): Forecasted return distribution
        spot (float)                        : Spot price
        strike (float)                      : Strike price
        risk_free_rate_optexp (float)       : Annual risk free rate for the period until option expiration
        cdays_optexp (float)                : Total number of calendar days until option expiration
        dvd_amts (list)                     : Dividend amounts
        risk_free_rates_exdvd (list)        : Annual risk free rate for the period until each ex-dividend date
        cdays_exdvds (list)                 : Total number of calendar days until ex-dividend for each dividend
        days_in_yr (float, optional)        : Total number of calendar days in the year (normal years: 365 days, leap years: 366)

    Returns:
        ((call_price, call_delta, borrow), (put_price, put_delta, loan))
    """
    
    # Fraction of year until option expiration
    t_optexp_calendar = cdays_optexp/days_in_yr
    
    # Adjust spot for discrete dividend
    spot_dvd_adj = adj_spot_for_dvd(spot, dvd_amts, risk_free_rates_exdvd, cdays_exdvds)

    # Normalize spot to $1 and strike correspondingly
    spot_norm = 1
    strike_norm = strike/spot_dvd_adj
    
    # Future prices
    future_prices = spot_norm*(future_return_distribution + 1)
    
    # Call and put payoffs from future prices
    call_payoffs = np.maximum(future_prices - strike_norm, 0)
    put_payoffs = np.maximum(strike_norm - future_prices, 0)
    
    # Function to minimize
    def _replicating_errors(parms):
        # Unpack optimization parameters
        call_delta, future_borrow, future_loan = parms

        # Replicating portfolio payoffs
        call_replicating_payoffs = future_prices*call_delta - future_borrow
        put_replicating_payoffs = future_prices*(call_delta - 1) + future_loan

        # Error between option payoffs and replicating payoffs
        call_errors = call_payoffs - call_replicating_payoffs
        put_errors = put_payoffs - put_replicating_payoffs

        # Mean squared error
        mse = (np.mean(call_errors**2) + np.mean(put_errors**2))/2

        return mse
    
    # Minimize
    res = minimize(
        fun=_replicating_errors,
        x0=[0.5, 0.5, 0.5],
        bounds=[(0,1), (0, max(2, strike_norm*2)), (0, max(2, strike_norm*2))],
        tol=1e-6,
    )

    if res.success:
        call_delta, future_borrow, future_loan = res.x
        put_delta = call_delta - 1
        
        discount_factor = np.exp(-risk_free_rate_optexp*t_optexp_calendar)
        borrow = future_borrow*discount_factor
        loan = future_loan*discount_factor
        
        # Set prices to present value of replicating portfolios
        call_price = spot_norm*call_delta - borrow
        put_price = spot_norm*put_delta + loan
        
        return (call_price*spot_dvd_adj, call_delta), (put_price*spot_dvd_adj, put_delta)
    else:
        return None