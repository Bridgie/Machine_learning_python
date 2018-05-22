import pandas as pd
import quandl, math
import numpy as np              #used for arrays
from sklearn import preprocessing, model_selection, svm
# 1. preprocessingfor for scalling data, accuracy, processing speed
# 2. cross_validation//model_selection since validation was deprecated for shuffling data, separate, timesaver
# 3.  svm for vector machines, doing regression
from sklearn.linear_model import LinearRegression
#imports regression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100       
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100    

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'          
df.fillna(-9999, inplace = True)      

forecast_out = int(math.ceil(0.01*len(df)))     # 3. shows how many days out it tries to forecast

df['label'] = df[forecast_col].shift(-forecast_out)   

df.dropna(inplace = True)

X = np.array(df.drop(['label'], 1))          #returns new df that drops label AKA features
y = np.array(df['label'])                     #returns new df that only contains labels converts it to array

X = preprocessing.scale(X)                   #Scales the data to one unit of numbers (Hard to explain)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
#shuffles the x's and y's, parameters are X, Y and 0.2 <- 20% of data will be used to test
#outputs train data and test data

clf  = LinearRegression()       #defines classifier
clf.fit(X_train, y_train)       #fits the classifier with training data

accuracy = clf.score(X_test, y_test)       #tests the classifier with testing data

print(accuracy)