# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:19:26 2018

@author: jason
"""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import re

"""
Extracts temperature, humidity, and wind speed from weather string

Parameters
----------
weather: string containing weather info for NFL game

Returns
----------
Pandas Series variables containing temperature, humidity, and wind speed

"""
def extractWeather(weather):
    weather = str(weather)
    tempMatch = re.match('([0-9]+)\sdegree',weather)
    humidMatch = re.match('.*humidity\s+([0-9]+)',weather)
    windMatch = re.match('.*wind\s+([0-9]+)\s+mph',weather)
    try:
        temp = int(tempMatch.group(1))
    except:
        temp = None
    try:
        humid = int(humidMatch.group(1))
    except:
        humid = None
    try:
        wind = int(windMatch.group(1))
    except:
        wind = None
    return pd.Series([temp,humid,wind])


"""
Extracts only the info from the team DataFrame that is needed for prediction
Parameters
----------
df: DataFrame of the team stats by game
"""
def transformTeamStats(df):
    df['rushYds'] = df['rush'].apply(lambda x: int(re.match("'*([0-9]+)-(-*[0-9]+)",x).group(2)))
    df['oppRushYds'] = df['oppRush'].apply(lambda x: int(re.match("'*([0-9]+)-(-*[0-9]+)",x).group(2)))
    df['carry'] = df['rush'].apply(lambda x: int(re.match("'*([0-9]+)-(-*[0-9]+)",x).group(1)))
    df['oppCarry'] = df['oppRush'].apply(lambda x: int(re.match("'*([0-9]+)-(-*[0-9]+)",x).group(1)))
    df['passYds'] = df['pass'].apply(lambda x: int(re.match("[0-9]+-([0-9]+)-(-*[0-9]+)",x).group(2)))
    df['oppPassYds'] = df['oppPass'].apply(lambda x: int(re.match("[0-9]+-([0-9]+)-(-*[0-9]+)",x).group(2)))
    df['passes'] = df['pass'].apply(lambda x: int(re.match("[0-9]+-([0-9]+)-(-*[0-9]+)",x).group(1)))
    df['oppPasses'] = df['oppPass'].apply(lambda x: int(re.match("[0-9]+-([0-9]+)-(-*[0-9]+)",x).group(1)))
    df['sacks']=df['sacks'].apply(lambda x:int(x.replace("'",'').split('-')[0]))
    df['oppSacks']=df['oppSacks'].apply(lambda x:int(x.replace("'",'').split('-')[0]))
    df['fumble'] = df['fumble'].apply(lambda x:int(x.replace("'",'').split('-')[0]))
    df['oppFumble'] = df['oppFumble'].apply(lambda x:int(x.replace("'",'').split('-')[0]))
    df['penalty'] = df['penalty'].apply(lambda x:int(x.replace("'",'').split('-')[0]))
    df['oppPenalty'] = df['oppPenalty'].apply(lambda x:int(x.replace("'",'').split('-')[0]))
    df['3rdDown'] = df['3rdDown'].apply(lambda x: x.replace("'",''))
    df['4thDown'] = df['4thDown'].apply(lambda x: x.replace("'",''))
    df['opp3rdDown'] = df['opp3rdDown'].apply(lambda x: x.replace("'",''))
    df['opp4thDown'] = df['opp4thDown'].apply(lambda x: x.replace("'",''))
    df['possession']=df['possession'].apply(lambda x: float(re.match('([0-9]+):([0-9]+)',x).group(1))+float(re.match('([0-9]+):([0-9]+)',x).group(2))/60)
    df['oppPossession']=df['oppPossession'].apply(lambda x: float(re.match('([0-9]+):([0-9]+)',x).group(1))+float(re.match('([0-9]+):([0-9]+)',x).group(2))/60)
    df['DownConversion'] = df[['3rdDown','4thDown']].apply(lambda x: (int(x[0].split('-')[0])+int(x[1].split('-')[0]))/(int(x[0].split('-')[1])+int(x[1].split('-')[1])),axis=1)
    df['oppDownConversion'] = df[['opp3rdDown','opp4thDown']].apply(lambda x: (int(x[0].split('-')[0])+int(x[1].split('-')[0]))/(int(x[0].split('-')[1])+int(x[1].split('-')[1])),axis=1)
    df.drop(['rush','pass','yds','4thDown','3rdDown'],axis=1,inplace=True)

def select(df,team):
#['date','week','team','opp_team']
    last4Games = team[df['team']].iloc[team[df['team']].index.get_loc(df['date'])-4:team[df['team']].index.get_loc(df['date'])]
    last4GamesOpps = team[df['opp_team']].iloc[team[df['opp_team']].index.get_loc(df['date'])-4:team[df['opp_team']].index.get_loc(df['date'])]
    if last4Games['week'].max() >= int(df['week']) or last4GamesOpps['week'].max() >= int(df['week']):
        print (last4Games)
        print (df['week'])
        print (df['date'])
        print (df['team'])
        print (team[df['team']].index.get_loc(df['date']))
        raise ValueError('ERROR: NOT ENOUGH DATA POINTS')
    else:
        average = pd.Series({'actual_score':team[df['team']].loc[df['date']]['score'],'score':last4Games['score'].mean(),'TO':last4Games['TO'].mean(),'1stDowns':last4Games['1stDowns'].mean(),'sacks':last4Games['sacks'].mean(),'fumble':last4Games['fumble'].mean(),'penalty':last4Games['penalty'].mean(),
        'possession':last4Games['possession'].mean(),'rushYds':last4Games['rushYds'].mean(),'passYds':last4Games['passYds'].mean(),'DownConversion':last4Games['DownConversion'].mean(),
        'rushEff':last4Games['rushYds'].sum()/last4Games['carry'].sum(),'passEff':last4Games['passYds'].sum()/last4Games['passes'].sum(),
        'oppScore':last4GamesOpps['oppScore'].mean(),'oppTO':last4GamesOpps['oppTO'].mean(),'opp1stDowns':last4GamesOpps['opp1stDowns'].mean(),'oppSacks':last4GamesOpps['oppSacks'].mean(),'oppFumble':last4GamesOpps['oppFumble'].mean(),'oppPenalty':last4GamesOpps['oppPenalty'].mean(),
        'oppPossession':last4GamesOpps['oppPossession'].mean(),'oppRushYds':last4GamesOpps['oppRushYds'].mean(),'oppPassYds':last4GamesOpps['oppPassYds'].mean(),'oppDownConversion':last4GamesOpps['oppDownConversion'].mean(),
        'oppRushEff':last4GamesOpps['oppRushYds'].sum()/last4GamesOpps['oppCarry'].sum(),'oppPassEff':last4GamesOpps['oppPassYds'].sum()/last4GamesOpps['oppPasses'].sum()})
        return average
#['date','week', 'score','yds','TO','1stDowns', 'rush', 'pass', 'sacks', 'fumble', 'penalty', '3rdDown', '4thDown', 'possession','oppSacks','oppScore','oppYds','oppTO']
"""
Instantiates new dataframe consisting of the game, team, the team's stats, and the opponent's teams stats
Returns only games that are past week 4 of the NFL season
Parameters
----------
df: DataFrame representing stats of home and away teams in a game
team: Dictionary consisting of DataFrames of team stats by game

Returns
----------
Dataframe consisting of games by team, game stats, stats of team in the last 4 games
stats of opponents in the last 4 games. DataFrame only consists of games past week 4 of NFL season
"""
def getGameStats(df, team):
    games = df[['date','week','home_team','away_team','roof','temp','humidity','wind']]
    games.rename(columns={'home_team':'team','away_team':'opp_team'}, inplace=True)
    
    awaygames = games
    awaygames['team'] = awaygames['opp_team']
    awaygames['opp_team'] = games['team']
    
    games = games.append(awaygames)
    games = games[games['week'].apply(int)>5]
    games.index=range(0,len(games.index))
    #Fill out the historical stats for games such as rolling 4 historical game averages
#    games[['actual_score','score','TO','1stDowns','sacks','fumble','penalty','possession','rushYds','passYds','DownConversion','rushEff','passEff','oppScore','oppTO','opp1stDowns','oppSacks','oppFumble','oppPenalty','oppPossession','oppRushYds','oppPassYds','oppDownConversion','oppRushEff','oppPassEff']] = games[['date','week','team','opp_team']].apply(select,axis=1,args=(team,))
    games = pd.concat([games,games[['date','week','team','opp_team']].apply(select,axis=1,args=(team,))],axis=1)
    return games

"""
Returns average temperature and humidity in NFL game by month

Parameters
----------
df: DataFrame containing date, temperature, and humidity
averageDF: Full DataFrame containing dates, temperature, and humidity to be used for averages
Returns
----------
Average temperature and humidity by month
"""
def fillTempHumidity(df, averageDF):
    tempHumid = [df['temp'],df['humidity']]
    byMonth = averageDF.groupby('month')['temp','humidity'].mean()
    if pd.isna(df['temp']):
            tempHumid[0] = byMonth.loc[int(df['month'])]['temp']
    if pd.isna(df['humidity']):
        tempHumid[1] = byMonth.loc[int(df['month'])]['humidity']
    return pd.Series(tempHumid)

"""
Clean NFL data

Parameters
----------
fulldf: DataFrame containing NFL games
----------
cleaned up data in DataFrame form
"""
def clean(fulldf):
    fulldf = pd.DataFrame()
    for i in range(2002,2018):
        temp = pd.read_csv('GameData/NFLgames'+str(i)+'.csv')
        fulldf = fulldf.append(temp)
    fulldf['date'] = pd.to_datetime(fulldf['date'])
    fulldf['week'] = fulldf['week'].apply(int)
    fulldf['home_score'] = fulldf['home_score'].apply(int)
    fulldf['away_score'] = fulldf['away_score'].apply(int)
    teams = {}
    
    for i in fulldf['home_team'].unique():
        teams.update({i:pd.DataFrame()})
    
    homeColumns = ['date','week', 'home_score', 'home_yds', 'home_TO', '1stDownsHome', 'rushHome', 'passHome', 'sacksHome', 'fumbleHome', 'penaltyHome', '3rdDownHome', '4thDownHome', 'posessionHome','possessionAway', 'sacksAway','away_score','away_yds','away_TO','1stDownsAway','fumbleAway','penaltyAway','rushAway', 'passAway', '3rdDownAway', '4thDownAway']
    awayColumns = ['date','week', 'away_score', 'away_yds', 'away_TO', '1stDownsAway', 'rushAway', 'passAway', 'sacksAway', 'fumbleAway', 'penaltyAway', '3rdDownAway', '4thDownAway', 'possessionAway','posessionHome', 'sacksHome','home_score','home_yds','home_TO','1stDownsHome','fumbleHome','penaltyHome','rushHome', 'passHome', '3rdDownHome', '4thDownHome']
    teamColumns = ['date','week', 'score','yds','TO','1stDowns', 'rush', 'pass', 'sacks', 'fumble', 'penalty', '3rdDown', '4thDown', 'possession','oppPossession','oppSacks','oppScore','oppYds','oppTO','opp1stDowns','oppFumble','oppPenalty','oppRush','oppPass','opp3rdDown','opp4thDown']
    
    for name, df in teams.items():
        teams[name] = fulldf[fulldf['home_team']==name][homeColumns]
        teams[name].columns = teamColumns
        teams[name]['home/away'] = 'home'
        away = fulldf[fulldf['away_team']==name][awayColumns]
        away.columns = teamColumns
        away['home/away'] = 'away'
        teams[name] = teams[name].append(away)
        teams[name].sort_values(by='date',ascending=True,inplace=True,axis=0)
        teams[name].set_index('date',inplace=True)
        transformTeamStats(teams[name])
    
    fulldf[['temp','humidity','wind']] = fulldf['weather'].apply(func=extractWeather)
    
    finalData = getGameStats(fulldf,teams)
    
    print (finalData.head(10))
    print (finalData.info())
    
    finalData['outdoors'] = finalData['roof'].apply(lambda x: 1 if (x=='outdoors' or x=='retractable roof (open)') else 0)
    #fillna for wind with no wind and fill nan for temperature with average temperature (53)
    print(finalData['wind'].fillna(0,inplace=True))
    finalData['windy'] = finalData['wind'].apply(lambda x: 1 if x>9.5 else 0)
    
    finalData['month'] = finalData['date'].dt.month
    
    #Fill out temperature and humidity using average per month
    finalData[['temp','humidity']] = finalData[['month','temp','humidity']].apply(fillTempHumidity,axis=1,args=(finalData[['month','temp','humidity']],))
    
    #Clean up rest of the dataset
    
    #is game being played late in season when NFL playoffs are generally already decided?
    finalData['lateinSeason'] = finalData['week'].apply(lambda x: 1 if x>14 else 0)
    
    #interaction effect between wind, outside, and pass yards
    finalData['windxpassYds'] = finalData['windy']*finalData['passYds']*finalData['outdoors']
    
    finalData.drop(columns = ['week','opp_team','roof','wind','month'],  inplace=True)
    return (finalData)