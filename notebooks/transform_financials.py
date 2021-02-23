# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python [conda env:neuralopt_env] *
#     language: python
#     name: conda-env-neuralopt_env-py
# ---

# +
# %load_ext autoreload
# %autoreload 2

import glob
import json
import os
import sys

import numpy as np
import pandas as pd


# -

# # Aggregate financial statements and save

def json_to_df(data):
    """Parses JSON data to a DataFrame
    """
    
    if data['timeseries']['error'] is None:
        data = data['timeseries']['result']
        
        dfs = []
        for field in data:
            field_key = field['meta']['type'][0]
            
            if 'trailing' not in field_key:
                
                if field_key in field:
                    df = pd.DataFrame([
                        (obs.get('asOfDate'), obs.get('reportedValue')['raw'])
                        for obs
                        in field[field_key]
                        if obs
                    ])
                    
                    field_key = field_key.replace('quarterly', '')
                    df.columns = ['Date', field_key]
                    df['Date'] =  pd.to_datetime(df['Date'])
                else:
                    field_key = field_key.replace('quarterly', '')
                    df = pd.DataFrame({'Date': [], field_key: []})
                    
                df = df.set_index('Date')
                dfs.append(df)
        
        if dfs:
            df = pd.concat(dfs, axis='columns').fillna(value=0)
            return df
    
    return None


files = glob.glob('../../../Investing Models/Data/Balance Sheets/*.json')
# files = np.random.choice(files, size=1000, replace=False)

for file in files:
    ticker = file.replace('../../../Investing Models/Data/Balance Sheets\\','').replace('.json', '')
    
    # Check if ticker exists in cashflows
    if os.path.exists(f'../../../Investing Models/Data/Cash Flows/{ticker}.json'): 
        statements = []
        for directory in [
            'Balance Sheets', 
            'Cash Flows'
        ]:

            with open(f'../../../Investing Models/Data/{directory}/{ticker}.json', 'r') as fp:
                data = json.load(fp)
                df = json_to_df(data)
                if df is not None:
                    statements.append(df)

        df_financials = pd.concat(statements, axis='columns').fillna(value=0)

        if len(df_financials) >= 5*4:
            df_financials.sort_index(axis=0).to_pickle(f'../../../Investing Models/Data/Financial Transforms/{ticker}.pkl')
