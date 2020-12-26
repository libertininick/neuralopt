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
import os
import random
import re
import time


from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests


# -

# # Scape earnings data

def scrape_earnings_date(date):
    
    # Create url from date
    url = f'https://finance.yahoo.com/calendar/earnings?day={date}'
    
    # Send initial request
    response = requests.get(url, timeout=(5,30))
    if response.status_code == 200:
        soup = BeautifulSoup(response.content)
        
        # Number of events
        n_events = 0
        m = soup.select_one('#fin-cal-table > h3 > span.Mstart\(15px\).Fw\(500\).Fz\(s\) > span')
        if m:
            m = re.findall(r'(?<=of )[0-9]{1,10}(?= results)', m.text)
            if m:
                n_events = int(m[0])
                
        if n_events > 0:
            try:
                # Extract first 100 results
                earnings_table = soup.select_one('#cal-res-table')
                earnings_table = pd.read_html(str(earnings_table))[0]
                
                # Page through other results
                for offset in range(100, n_events, 100):
                    # Sleep for 2 seconds
                    time.sleep(2)
                    
                    # Add offset to URL
                    url_i = f'{url}&offset={offset}&size=100'
                    
                    # Send ith request
                    response = requests.get(url_i, timeout=(5,30))
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content)
                        try:
                            earnings_table_i = soup.select_one('#cal-res-table')
                            earnings_table_i = pd.read_html(str(earnings_table_i))[0]
                            if len(earnings_table_i):
                                earnings_table = pd.concat([earnings_table, earnings_table_i], axis='rows')
                        except:
                            pass
                        
                return earnings_table
                
            except:
                return None


# ## Scrape every date

# +
dates = pd.date_range(
    start='1994-12-31', 
    end=datetime.today(), 
    freq='D'
)
dates = [dt.strftime('%Y-%m-%d') for dt in dates]
print(f'Number of dates: {len(dates):,}')


for i, dt in enumerate(dates, 1):
    result = scrape_earnings_date(dt)
    if result is not None:
        # Save
        result.to_csv(f'../../../Investing Models/Earnings data/{date}.csv')
        
        # Randomly sleep
        time.sleep(random.randint(0,10))
        
        # Print progress
        print(f'{i/len(dates):>6.2%} {date}')
# -

# # Compile earnings data

# +
dfs = []
for f in glob.glob('../../../Investing Models/Earnings data/*.csv'):
    m = re.findall(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', f)
    if m:
        date = m[0]
        
        df = pd.read_csv(f)
        df['Date'] = date

        df = df[['Date', 'Company', 'Symbol', 'EPS Estimate', 'Reported EPS']]
        df = df.groupby('Company').apply(lambda g: g.sort_values(by='Symbol').iloc[0])
        df.index = np.arange(len(df))
        
        dfs.append(df)
        
all_earnings_data = pd.concat(dfs, axis='rows')

# Save
all_earnings_data.to_csv(
    f'../../../Investing Models/Earnings data/all_earnings_data.csv',
    index=False
)


# -

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
                f'../../../Investing Models/Price data/{ticker}.csv',
                index=False
            )
            return True
    except:
        pass
    
    return False


# ## Compile tickers from earnings dates

# +
df_earnings = pd.read_csv(f'../../../Investing Models/Earnings data/all_earnings_data.csv')
tickers = np.unique(df_earnings['Symbol'])
print(f'All tickers: {len(tickers):,}')

# Data already saved
tickers_old = [
    f.replace('../../../Investing Models/Price data\\', '').replace('.csv', '')
    for f 
    in glob.glob('../../../Investing Models/Price data/*.csv')
]

print(f'Old tickers: {len(tickers_old):,}')

# Difference
tickers_new = set(tickers).difference(tickers_old)
print(f'New tickers: {len(tickers_new):,}')
# -

# ## Scrape new tickers

for ticker in tickers_new:
    result = scrape_price_data(ticker, '1995-12-31', '2020-11-30')

    # Randomly sleep
    time.sleep(random.randint(0,10))
    
    print(ticker, result)


