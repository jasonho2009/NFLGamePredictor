# NFL Game Predictor

Used Python and the BeautifulSoup library to web scrape NFL game data from all regular NFL football games since 2002 (over 4000 games). Data is saved as csv file by year

Web scraper scrapes all game data including points scored, home/away teams, where game is played, passing yards, rushing yards, sacks, and even weather. 

Once data was scraped, used Linear Regression, Ridge Regression, and Lasso on the data to predict how each team would score in each game. Using this predicted score, 
predict which team would win, by how much (betting spread), and total points scored (over/under betting)

Overall, 55% accuracy of correctly predicting the winner and 60% accuracy of correctly predicting over/under and betting spread