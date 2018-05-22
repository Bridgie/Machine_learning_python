import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100       
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100    

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'           #whatever you are trying to forecast
df.fillna(-9999, inplace = True)      #needed to fill out any missing data, Algorithm needs Data and nothing else.

forecast_out = int(math.ceil(0.01*len(df)))      # 1. math.ceil will return a float but will round the number to highest in
                                                # 2. 0.1 is used because we are trying to forecast the nearest 10%


df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace = True)
print(df.head())