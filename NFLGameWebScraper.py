# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:30:17 2018

@author: jason
"""

import requests
from bs4 import BeautifulSoup
import re
import csv
import time

def specificGameData(url):
    print(url)
    response = requests.get('https://www.pro-football-reference.com'+str(url),timeout=200)
    html = response.text
    soup = BeautifulSoup(html,"html.parser")
    game_info = soup.find_all(string=re.compile('Game Info Table'))[0]
    info_soup = BeautifulSoup(game_info,'html.parser')
    teamStats = soup.find_all(string=re.compile('Team Stats Table'))[0]
    team_soup = BeautifulSoup(teamStats,'html.parser')
    try:
        try:
            weather = info_soup.find('th',string='Weather').nextSibling.contents[0]
        except:
            weather = ''
        environment = [info_soup.find('th',string='Roof').nextSibling.contents[0],info_soup.find('th',string='Surface').nextSibling.contents[0], weather,info_soup.find('th',string='Vegas Line').nextSibling.contents[0],info_soup.find('th',string='Over/Under').nextSibling.contents[0]]
        homeStats = []
        visStats = []
        homeStats.append(team_soup.find_all('th',string='First Downs')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append(team_soup.find_all('th',string='First Downs')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append(team_soup.find_all('th',string='Rush-Yds-TDs')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append(team_soup.find_all('th',string='Rush-Yds-TDs')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append(team_soup.find_all('th',string='Cmp-Att-Yd-TD-INT')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append(team_soup.find_all('th',string='Cmp-Att-Yd-TD-INT')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append("'"+team_soup.find_all('th',string='Sacked-Yards')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append("'"+team_soup.find_all('th',string='Sacked-Yards')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append("'"+team_soup.find_all('th',string='Fumbles-Lost')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append("'"+team_soup.find_all('th',string='Fumbles-Lost')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append("'"+team_soup.find_all('th',string='Penalties-Yards')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append("'"+team_soup.find_all('th',string='Penalties-Yards')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append("'"+team_soup.find_all('th',string='Third Down Conv.')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append("'"+team_soup.find_all('th',string='Third Down Conv.')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append("'"+team_soup.find_all('th',string='Fourth Down Conv.')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append("'"+team_soup.find_all('th',string='Fourth Down Conv.')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
    
        homeStats.append(team_soup.find_all('th',string='Time of Possession')[0].parent.find(attrs={'data-stat':'home_stat'}).contents[0])
        visStats.append(team_soup.find_all('th',string='Time of Possession')[0].parent.find(attrs={'data-stat':'vis_stat'}).contents[0])
        
        return environment, homeStats, visStats
    except:
        print(info_soup)
        print (team_soup)
#for i in range(2002,2017)
def printCSV(table,writer):
    for tr in table.find_all('tr'):
        if 'PLAYOFFS' in str(tr.contents).upper():
            return
        if (tr.find(attrs={'data-stat':'week_num'}).contents[0]).upper() == 'WEEK':
            pass
        else:
            week = tr.find(attrs={'data-stat':'week_num'}).contents[0]
            date = tr.find(attrs={'data-stat':'game_date'})['csk']
            if not tr.find(attrs={'data-stat':'game_location'}).contents:
                home_team = tr.find(attrs={'data-stat':'winner'}).find_all('a')[0].contents[0]
                away_team = tr.find(attrs={'data-stat':'loser'}).find_all('a')[0].contents[0]
                if 'strong' in str(tr.find(attrs={'data-stat':'pts_win'}).contents[0]):
                    home_score = tr.find(attrs={'data-stat':'pts_win'}).contents[0].contents[0]
                else:
                    home_score = tr.find(attrs={'data-stat':'pts_win'}).contents[0]
                
                if 'strong' in str(tr.find(attrs={'data-stat':'pts_lose'}).contents[0]):
                    away_score = tr.find(attrs={'data-stat':'pts_lose'}).contents[0].contents[0]
                else:
                    away_score = tr.find(attrs={'data-stat':'pts_lose'}).contents[0]
                
                home_yds = tr.find(attrs={'data-stat':'yards_win'}).contents[0]
                away_yds = tr.find(attrs={'data-stat':'yards_lose'}).contents[0]
                home_TO = tr.find(attrs={'data-stat':'to_win'}).contents[0]
                away_TO = tr.find(attrs={'data-stat':'to_lose'}).contents[0]
            else:
                away_team = tr.find(attrs={'data-stat':'winner'}).find_all('a')[0].contents[0]
                home_team = tr.find(attrs={'data-stat':'loser'}).find_all('a')[0].contents[0]
                
                if 'strong' in str(tr.find(attrs={'data-stat':'pts_win'}).contents[0]):
                    away_score = tr.find(attrs={'data-stat':'pts_win'}).contents[0].contents[0]
                else:
                    away_score = tr.find(attrs={'data-stat':'pts_win'}).contents[0]
                
                if 'strong' in str(tr.find(attrs={'data-stat':'pts_win'}).contents[0]):
                    home_score = tr.find(attrs={'data-stat':'pts_win'}).contents[0].contents[0]
                else:
                    home_score = tr.find(attrs={'data-stat':'pts_win'}).contents[0]
                away_yds = tr.find(attrs={'data-stat':'yards_win'}).contents[0]
                home_yds = tr.find(attrs={'data-stat':'yards_lose'}).contents[0]
                away_TO = tr.find(attrs={'data-stat':'to_win'}).contents[0]
                home_TO = tr.find(attrs={'data-stat':'to_lose'}).contents[0]
            environment, homeStats, visStats = specificGameData(tr.find(attrs={'data-stat':'boxscore_word'}).find('a')['href'])
            writer.writerow({'date':date,'week':week,'home_team':home_team,'away_team':away_team,'home_score':home_score,
                             'away_score':away_score,'home_yds':home_yds,'away_yds':away_yds,'home_TO':home_TO,'away_TO':away_TO,'roof':environment[0],'surface':environment[1],'weather':environment[2],
                             '1stDownsHome':homeStats[0],'1stDownsAway':visStats[0],'rushHome':homeStats[1],'rushAway':visStats[1],'passHome':homeStats[2],'passAway':visStats[2],
                             'sacksHome':homeStats[3],'sacksAway':visStats[3],'fumbleHome':homeStats[4],'fumbleAway':visStats[4],'penaltyHome':homeStats[5],'penaltyAway':visStats[5],
                             '3rdDownHome':homeStats[6],'3rdDownAway':visStats[6],'4thDownHome':homeStats[7],'4thDownAway':visStats[7],'posessionHome':homeStats[8],'possessionAway':visStats[8],'VegasLine':environment[3],'Over/Under':environment[4]})  

def getHistory(year): 
    for year in years:
        print (year)
        with open('GameData/NFLgames'+str(year)+'.csv','w') as csvfile:
            fieldnames = ['date','week','home_team','away_team','home_score','away_score','home_yds','away_yds','home_TO','away_TO','roof','surface','weather','1stDownsHome','1stDownsAway'
                          ,'rushHome','rushAway','passHome','passAway','sacksHome','sacksAway','fumbleHome','fumbleAway','penaltyHome','penaltyAway','3rdDownHome','3rdDownAway'
                          ,'4thDownHome','4thDownAway','posessionHome','possessionAway','VegasLine','Over/Under']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
            writer.writeheader()
            
            response = requests.get('https://www.pro-football-reference.com/years/'+str(year)+'/games.htm')
            html = response.text
            soup = BeautifulSoup(html,"html.parser")
            table = soup.find(name='table',id='games')
            printCSV(table,writer)
            
years = range(2002,2018)
getHistory(years)