# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:56:55 2018

@author: jason
"""
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

import NFLDataCleanUp


fulldf = pd.DataFrame()
for i in range(2002,2018):
    temp = pd.read_csv('GameData/NFLgames'+str(i)+'.csv')
    fulldf = fulldf.append(temp)
NFLdata = NFLDataCleanUp.clean(fulldf)

NFLdata = NFLdata[['actual_score','date','team','temp', 'humidity', '1stDowns', 'DownConversion', 'TO',
       'fumble', 'opp1stDowns', 'oppDownConversion', 'oppFumble', 'oppPassEff',
       'oppPassYds', 'oppPenalty', 'oppPossession', 'oppRushEff', 'oppRushYds',
       'oppSacks', 'oppScore', 'oppTO', 'passEff', 'passYds', 'penalty',
       'possession', 'rushEff', 'rushYds', 'sacks', 'score', 'outdoors',
       'windy', 'lateinSeason', 'windxpassYds']]

#training data, test data split
X_trainWithDate, X_testWithDate, y_train, y_test = train_test_split(NFLdata.drop(columns=['actual_score']), NFLdata['actual_score'], test_size=0.3, random_state=42)

X_train = X_trainWithDate.drop(columns=['date','team'])
X_test = X_testWithDate.drop(columns=['date','team'])

lr = LinearRegression()
lr.fit(X_train,y_train)

predictedValues = lr.predict(X_test)
predictedValues = np.maximum(predictedValues,np.zeros(predictedValues.shape))
print ('Mean Absolute Error: ',((predictedValues-y_test).abs()).mean())
print ('Root Mean Squared Error: ',math.sqrt(((predictedValues-y_test).apply(np.square)).mean()))

fulldf['date'] = pd.to_datetime(fulldf['date'])
fulldf['Over/Under'] = fulldf['Over/Under'].apply(int)

predictedValuesTeam = X_testWithDate[['date','team',]]
predictedValuesTeam['predictedScore'] = predictedValues

fulldf = fulldf[['date','home_team','away_team','Over/Under']]

def getPredictedScore(df, predictions):
    try:
        return (predictions[(predictions[predictions['date']==df['date']]) & (predictions[predictions['team']==df['home_team']])]['predictedScore']+
                        predictions[(predictions[predictions['date']==df['date']]) & (predictions[predictions['team']==df['away_team']])]['predictedScore'])
    except:
        print (df['date'])
        print (predictions[predictions['date']==df['date']])
        print (predictions[predictions['team']==df['home_team']])
        return None
fulldf.to_csv('testfulldf.csv')
predictedValuesTeam.to_csv('testPredicted.csv')
fulldf['predicted'] = fulldf.apply(getPredictedScore,axis=1,args=(predictedValuesTeam,))
print (fulldf[~(fulldf['predicted']==None)])
print (fulldf['predicted'].unique())