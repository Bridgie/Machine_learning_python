import pandas as pd
import quandl, math, datetime
import numpy as np              
from sklearn import preprocessing, model_selection, svm

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
#plots stuff
from matplotlib import style
#makes plotted stuff look decent
style.use('ggplot')
#which style to use

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100       
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100   


df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'           
df.fillna(-9999, inplace = True)      

forecast_out = int(math.ceil(0.1*len(df)))     

df['label'] = df[forecast_col].shift(-forecast_out)   



X = np.array(df.drop(['label','Adj. Close'], 1))         

X = preprocessing.scale(X)                 
X_lately = X[-forecast_out:]             #The thing we are predicting against, think y = mx + b and we have x
                                         #we need to figure out the m and b, we get answer for y
                                         #AKA THE LAST 30 DAYS OF DATA
X = X[:-forecast_out]                       # [:] slices the array at that given index


df.dropna(inplace = True)

y = np.array(df['label'])     
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)


clf  = LinearRegression()       
clf.fit(X_train, y_train)      

accuracy = clf.score(X_test, y_test)       
forecast_set = clf.predict(X_lately)       #predicts using the last 30 days of data form X_lately


df['Forecast'] = np.nan

#hardcoded for plotting since we dont have dates as features
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


for i in forecast_set:              #iterates thru the forecast set, taking each forecast and day
    next_date = datetime.datetime.fromtimestamp(next_unix)  
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    #then setting those to the values in the DF, making the future features NAN
    #finally takes the first columns and sets them to NAN
    #and the final column is the forecast

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()