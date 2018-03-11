# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:56:55 2018

@author: jason
"""
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import re

import NFLDataCleanUp


fulldf = pd.DataFrame()
for i in range(2002,2018):
    temp = pd.read_csv('GameData/NFLgames'+str(i)+'.csv')
    fulldf = fulldf.append(temp)
train, test  = NFLDataCleanUp.clean(fulldf, 0.3)

train = train[['actual_score','date','team','temp', 'humidity', '1stDowns', 'DownConversion', 'TO',
       'fumble', 'opp1stDowns', 'oppDownConversion', 'oppFumble', 'oppPassEff',
       'oppPassYds', 'oppPenalty', 'oppPossession', 'oppRushEff', 'oppRushYds',
       'oppSacks', 'oppScore', 'oppTO', 'passEff', 'passYds', 'penalty',
       'possession', 'rushEff', 'rushYds', 'sacks', 'score', 'outdoors',
       'windy', 'lateinSeason', 'windxpassYds']]

test = test[['actual_score','date','team','temp', 'humidity', '1stDowns', 'DownConversion', 'TO',
       'fumble', 'opp1stDowns', 'oppDownConversion', 'oppFumble', 'oppPassEff',
       'oppPassYds', 'oppPenalty', 'oppPossession', 'oppRushEff', 'oppRushYds',
       'oppSacks', 'oppScore', 'oppTO', 'passEff', 'passYds', 'penalty',
       'possession', 'rushEff', 'rushYds', 'sacks', 'score', 'outdoors',
       'windy', 'lateinSeason', 'windxpassYds']]


#training data, test data split
X_trainWithDate = train.drop(columns=['actual_score'])
y_train = train['actual_score']
X_testWithDate = test.drop(columns=['actual_score'])
y_test = test['actual_score']

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

fulldf = fulldf[['date','home_team','away_team','Over/Under','home_score','away_score','VegasLine']]

def VegasLineExpression(df):
    if df['VegasLine'] == 'Pick':
        homeOrAway = 'home'
        winBy = 0
    else:
        match = re.match('(.*) -([0-9.]*)',df['VegasLine'])
        team = match.group(1)
        winBy = float(match.group(2))
        homeOrAway = ''
        if team == df['home_team']:
            homeOrAway = 'home'
        elif team == df['away_team']:
            homeOrAway = 'away'
        else:
            raise ValueError('not matching teams in Vegas Line Expression')
        
    return pd.Series([homeOrAway,winBy])

fulldf['home_score'] = fulldf['home_score'].apply(int)
fulldf['away_score'] = fulldf['away_score'].apply(int)
fulldf['Over/Under'] = fulldf['Over/Under'].apply(int)
fulldf[['VegasHomeOrAway','Spread']] = fulldf.apply(VegasLineExpression,axis=1)


def getPredictedScore(df, predictions):
    date = df['date']
    home_team = df['home_team']
    away_team = df['away_team']
    home_score = 0
    away_score = 0
    try:
        if (len(predictions[(predictions['date']==date) & (predictions['team']==home_team)]['predictedScore'].index) <= 1):
            home_score =  predictions[(predictions['date']==date) & (predictions['team']==home_team)]['predictedScore'].mean()
        else:
            raise ValueError('More than 1 matches')
        if (len(predictions[(predictions['date']==date) & (predictions['team']==away_team)]['predictedScore'].index) <=1):
            away_score = predictions[(predictions['date']==date) & (predictions['team']==away_team)]['predictedScore'].mean()
        else:
            raise ValueError('More than 1 matches')
        return pd.Series([home_score, away_score])
    except:
        return pd.Series([None,None])
fulldf.to_csv('testfulldf.csv')
predictedValuesTeam.to_csv('testPredicted.csv')
fulldf[['pred_homeScore','pred_awayScore']] = fulldf.apply(getPredictedScore,axis=1,args=(predictedValuesTeam,))

fulldf = fulldf[fulldf['pred_homeScore'].notnull()]

def predicting(df):
    overunder = False
    win = False
    spread = False
    if df['home_score']+df['away_score'] >= df['Over/Under'] and df['pred_homeScore']+df['pred_awayScore'] >= df['Over/Under']:
        overunder= True
    elif df['home_score']+df['away_score'] <= df['Over/Under'] and df['pred_homeScore']+df['pred_awayScore'] <= df['Over/Under']:
        overunder = True
    else:
        overunder = False
        
    if df['home_score'] >= df['away_score'] and df['pred_homeScore'] >= df['pred_awayScore']:
        win = True
    elif df['home_score'] <= df['away_score'] and df['pred_homeScore'] <= df['pred_awayScore']:
        win = True
    else:
        win = False
    if df['Spread'] == 0:
        if abs(df['pred_homeScore'] - df['pred_awayScore']) < 1:
            spread = True
        else:
            spread= False
    else:
        if df['VegasHomeOrAway'] =='home':
            if (df['pred_homeScore'] - df['pred_awayScore']) >= df['Spread']:
                if (df['home_score']-df['away_score']) >= df['Spread']:
                    spread = True
                else:
                    spread = False
            else:
                if (df['home_score']-df['away_score']) < df['Spread']:
                    spread = True
                else:
                    spread = False
        elif df['VegasHomeOrAway'] =='away':
            if (df['pred_awayScore'] - df['pred_homeScore']) >= df['Spread']:
                if (df['away_score']-df['home_score']) >= df['Spread']:
                    spread = True
                else:
                    spread = False
            else:
                if (df['away_score']-df['home_score']) < df['Spread']:
                    spread = True
                else:
                    spread = False
        else:
            raise ValueError('Neither home nor away')
    return pd.Series([overunder,win,spread])
    
fulldf[['pred_over/under','win','pred_spread']] = fulldf.apply(predicting,axis=1)

print ('Correct % of Over/Under predicted: ',fulldf[fulldf['pred_over/under']==True]['pred_over/under'].count()/fulldf['pred_over/under'].count()*100)

print ('Correct % of Game outcomes predicted: ',fulldf[fulldf['win']==True]['win'].count()/fulldf['win'].count()*100)

print ('Correct % of spread outcomes predicted: ',fulldf[(fulldf['pred_spread']==True) & (fulldf['Spread']!=0)]['pred_spread'].count()/fulldf[fulldf['Spread']!=0]['pred_spread'].count()*100)

