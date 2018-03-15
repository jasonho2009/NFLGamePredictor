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
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import math
import re

import NFLDataCleanUp


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
        
    if df['home_score'] > df['away_score'] and df['pred_homeScore'] > df['pred_awayScore']:
        win = True
    elif df['home_score'] < df['away_score'] and df['pred_homeScore'] < df['pred_awayScore']:
        win = True
    elif df['home_score'] == df['away_score'] and abs(df['pred_homeScore'] - df['pred_awayScore']) <=1.5:
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


fulldf['date'] = pd.to_datetime(fulldf['date'])
fulldf['Over/Under'] = fulldf['Over/Under'].apply(int)

fulldf = fulldf[['date','home_team','away_team','Over/Under','home_score','away_score','VegasLine']]


fulldf['home_score'] = fulldf['home_score'].apply(int)
fulldf['away_score'] = fulldf['away_score'].apply(int)
fulldf['Over/Under'] = fulldf['Over/Under'].apply(int)
fulldf[['VegasHomeOrAway','Spread']] = fulldf.apply(VegasLineExpression,axis=1)

#training data, test data split
X_trainWithDate = train.drop(columns=['actual_score'])
y_train = train['actual_score']
X_testWithDate = test.drop(columns=['actual_score'])
y_test = test['actual_score']

X_train = X_trainWithDate.drop(columns=['date','team'])
X_test = X_testWithDate.drop(columns=['date','team'])

lr = LinearRegression()
lr.fit(X_train,y_train)

linearRegPredictions = X_testWithDate[['date','team',]]
linearRegPredictions['predictedScore'] = np.maximum(lr.predict(X_test),np.zeros((lr.predict(X_test)).shape))


#10-fold cross validation to find optimal lambda for Ridge Regression
kf = KFold(n_splits=10,shuffle=True)
mseArray = np.array([])
for alpha in np.arange(0.1, 1, 0.1):
    ridge = Ridge(alpha=alpha, normalize = True)
    rmse = 0
    for train_index, test_index in kf.split(X_train):
        ridge.fit(X_train.loc[train_index],y_train.loc[train_index])
        pred = ridge.predict(X_train.loc[test_index])
        rmse += math.sqrt(((pred-y_train.loc[test_index]).apply(np.square)).mean())
    rmse = rmse/10
    mseArray = np.append(mseArray, rmse)
np.set_printoptions(threshold=50)
plt.plot(np.arange(0.1,1,0.1),mseArray)
print (mseArray.min())
print (np.argmin(mseArray))

#10-fold cross validation to find optimal lambda for Lasso Regression
kf = KFold(n_splits=10,shuffle=True)
mseArray = np.array([])
for alpha in np.arange(0.1, 1, 0.1):
    ridge = Lasso(alpha=alpha, normalize = True)
    rmse = 0
    for train_index, test_index in kf.split(X_train):
        ridge.fit(X_train.loc[train_index],y_train.loc[train_index])
        pred = ridge.predict(X_train.loc[test_index])
        rmse += math.sqrt(((pred-y_train.loc[test_index]).apply(np.square)).mean())
    rmse = rmse/10
    mseArray = np.append(mseArray, rmse)
np.set_printoptions(threshold=50)
plt.plot(np.arange(0.1,1,0.1),mseArray)
print (mseArray.min())
print (np.argmin(mseArray))

ridge = Ridge(alpha = 0.8, normalize = True)
ridge.fit(X_train,y_train)

ridgePredictions = X_testWithDate[['date','team',]]
ridgePredictions['predictedScore'] = np.maximum(ridge.predict(X_test),np.zeros((ridge.predict(X_test)).shape))

lasso = Lasso(alpha = 0.8, normalize = True)
lasso.fit(X_train,y_train)

lassoPredictions = X_testWithDate[['date','team',]]
lassoPredictions['predictedScore'] = np.maximum(lasso.predict(X_test),np.zeros((lasso.predict(X_test)).shape))
pd.set_option('display.max_columns', 50)

def EvaluatePredictions(allDF, method, predictedValuesTeam, y_test, coefficients,columns, scaler):
    allGames = allDF.copy(deep=True)
    print ('--------------'+str(method)+' Results'+'-----------------')
    allGames[['pred_homeScore','pred_awayScore']] = allGames.apply(getPredictedScore,axis=1,args=(predictedValuesTeam,))
    allGames = allGames[allGames['pred_homeScore'].notnull()]  
    allGames[['pred_over/under','win','pred_spread']] = allGames.apply(predicting,axis=1)

    allGames.to_csv(method+'_Results'+'.csv')
    print(pd.DataFrame(data=(coefficients/scaler.mean_).reshape(1,len(columns)),index=['scaled_coefficients'],columns=columns))
    
    
    print ('Mean Absolute Error: {:.2f}'.format(((predictedValuesTeam['predictedScore']-y_test).abs()).mean()))
    print ('Root Mean Squared Error: {:.2f}'.format(math.sqrt(((predictedValuesTeam['predictedScore']-y_test).apply(np.square)).mean())))
    print ('Correct % of Over/Under predicted: {:.2f}%'.format(allGames[allGames['pred_over/under']==True]['pred_over/under'].count()/allGames['pred_over/under'].count()*100))
    print ('Correct % of Game outcomes predicted: {:.2f}%'.format(allGames[allGames['win']==True]['win'].count()/allGames['win'].count()*100))
    print ('Correct % of spread outcomes predicted: {:.2f}%'.format(allGames[(allGames['pred_spread']==True) & (allGames['Spread']!=0)]['pred_spread'].count()/allGames[allGames['Spread']!=0]['pred_spread'].count()*100))

scaler = StandardScaler(with_mean=False)
scaler.fit(X_train)
print(pd.DataFrame(data=(scaler.mean_).reshape(1,len(X_train.columns)),index=['std_dev_scale'],columns=X_train.columns))

EvaluatePredictions(fulldf, "Linear_Regression", linearRegPredictions, y_test, lr.coef_,X_train.columns, scaler)
EvaluatePredictions(fulldf, "Ridge_Regularization", ridgePredictions, y_test, ridge.coef_,X_train.columns, scaler)
EvaluatePredictions(fulldf, "Lasso", lassoPredictions, y_test, lasso.coef_,X_train.columns, scaler)