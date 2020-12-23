# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: neuralopt_env
#     language: python
#     name: neuralopt_env
# ---

# +
from datetime import datetime
import glob
import random
import re
import time


from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests


# -

# # Scape earnings data

def scrape_earnings_calendar(date):
    url = f'https://finance.yahoo.com/calendar/earnings?day={date}'

    response = requests.get(url, timeout=(5,30))
    if response.status_code == 200:
        soup = BeautifulSoup(response.content)
        
    try:
        earnings_table = soup.select_one('#cal-res-table')
        earnings_table = pd.read_html(str(earnings_table))[0]
        
        if len(earnings_table):
            earnings_table.to_csv(f'C:/Users/liber/Dropbox/Investing Models/Earnings data/{date}.csv')
            return True
    except:
        pass
    
    return False


# +
dates = pd.date_range(
    start='1999-12-31', 
    end='2010-12-31',
#     end=datetime.today(), 
    freq='D'
)
dates = [dt.strftime('%Y-%m-%d') for dt in dates if dt.weekday() < 5]
print(f'Number of dates: {len(dates):,}')

issues = []
for dt in dates:
    result = scrape_earnings_calendar(dt)
    
    print(dt, result)
    if not result:
        issues.append(dt)

    # Randomly sleep
    time.sleep(random.randint(0,10))

print()
print(f'Number of issues: {len(issues):,}')
print(issues)
# -

# # Compile earnings data

dfs = []
for f in glob.glob('C:/Users/liber/Dropbox/Investing Models/Earnings data/*.csv'):
    m = re.findall(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', f)
    if m:
        date = m[0]
        
        df = pd.read_csv(f)
        df['Date'] = date

        df = df[['Date', 'Company', 'Symbol', 'EPS Estimate', 'Reported EPS']]
        df = df.groupby('Company').apply(lambda g: g.sort_values(by='Symbol').iloc[0])
        df.index = np.arange(len(df))
        
        dfs.append(df)

data = pd.concat(dfs, axis='rows')
data.to_csv(
    f'C:/Users/liber/Dropbox/Investing Models/Earnings data/all_data.csv',
    index=False
)


# # Scrape price data

def scrape_price_data(ticker, date_from, date_to):
    reference_date = '1995-12-30'
    reference_period = 820368000
    day_scale = 86400
    
    dt0 = datetime.strptime(reference_date, '%Y-%m-%d')
    dt1 = datetime.strptime(date_from, '%Y-%m-%d')
    dt2 = datetime.strptime(date_to, '%Y-%m-%d')
    
    period1 = (dt1 - dt0).days*day_scale + reference_period
    period2 = (dt2 - dt0).days*day_scale + reference_period

    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
    try:
        price_table = pd.read_csv(url)
        if len(price_table):
            price_table['Symbol'] = ticker
            price_table.to_csv(
                f'C:/Users/liber/Dropbox/Investing Models/Price data/{ticker}.csv',
                index=False
            )
            return True
    except:
        pass
    
    return False


# +
tickers = np.unique(data['Symbol'])
print(f'Number of tickers: {len(tickers):,}')

issues = []
for ticker in tickers:
    result = scrape_price_data(ticker, '1995-12-31', '2020-11-30')
    
    print(ticker, result)
    if not result:
        issues.append(ticker)

    # Randomly sleep
    time.sleep(random.randint(0,10))

print()
print(f'Number of issues: {len(issues):,}')
# -


