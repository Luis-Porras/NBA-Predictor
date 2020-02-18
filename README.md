
# NBA Predictor

The "NBA Predictor" is a python project which calcuates whether a home team would win (1) or lose (0) a given matchup using the sklearn machine learning library. 


# Data Collection

Basketball-reference.com keeps historical NBA data such as match results, full team rosters, individual player stats, and detailed league leader stats as free, downloadable .CSV files. 
https://www.basketball-reference.com/

The NBA Predictor uses a single NBA season at a time to train and test the models. The seasons used were the 2016-17, 2017-18, 2018-19, and 2019-20 seasons. 

# Data Preparation (Cleaning, Feature Selection, Fitting model)

***Cleaning***
The .CSV files from basketball-reference.com are nearly machine-learning-ready datasets. First, dropped unecessary columns such as date of game, multiple "unnamed" columns with many null values, and attendance count. The only columns kept were the Home, Away, and HomeWin columns. Used dummy variables to one-hot encode team names. 

***Feature Engineering and Selection***
Research from multiple sports gambling websites shows that professional handicappers look at winning streaks, losing streaks, team talent, and potential fatigue from too many consecutive away games as some of the most important influences for their NBA bets. I recreated these features using the original data set. Reference functions in main_scripts/main.ipynb: timeaway(df1,df2), allstar_count(df1,df2), win_streak(df1,df2), lose_streak(df1,df2)

Some other features were tested but ultimately left out due to the difficulty of recreating the features across multiple seasons. Examples:

__Bench Rating__: Teams with better lineup depth should perform better than teams with only a few key players. Only tried this on one dataset. This feature would require mutliple hours of parsing through team rosters to check which players may be starters, and which players may be on the bench. A thorough web scrape is necessary to collect the bench player rosters alone. After saving the bench roster, the individual player ratings were collected from NBA 2K's player rosters. This is video game data, which is fairly static across a season, and also not necessarily an accurate measure of a player's performance. 

Future Bench Rating fix: 
-Web scrape for bench rosters.
-Should create a way to dynamically quantify a player's performance rating without using video game data.

__Coach Rating__: This feature gave a rating from 1-5 to the 5 teams with the "best" coaches. All other teams received a zero in this column. Research was done by checking historical coach winnings, reddit forums, ranker.com, and watching youtube videos. This seemed a little too opinionated to include in the model.

__Average score__: This is still feasible, but currently has not been a good fit for the models being used. I believe this is a collinearity problem, as this column's correlation value to some of the other features was significantly higher than preferred. 


# Model Fitting and Evaluation 

All models used are supervised machine learning models. 

Models: Logistic Regression, Random Forest Classifier, Support Vector Classifier
Final Features: allstar count, losing streak, winning streak, time away 
Target: Home win (1, 0) 

80% Training data (980 games) 
20% Testing data (250 games) 

***Accuracy Scores***

|                           | 2016/17  | 2017/18  | 2018/19  | 2019/20 |
|---------------------------|----------|----------|----------|---------|
| Logistic Regression       | 61.5%    | 69.4%    | 68.7%    | 69.6%   |
| Random Forest Classifier  | 55.5%    | 69.4%    | 63.8%    | 71.7%   |
| Support Vector Classifier | 65.2%    | 66.1%    | 61.4%    | 59.8%   |


Currently, the training data is the first 980 games of a season, and the testing data is the last 250 games of a season. This project definitely needs cross validation to correctly evaluate an accuracy score. 

Scaling features seems to cause some accuracy problems, but scaling function is created in main_scripts/main.py. (scale(features, df))

# Picking Winners/ Gambling Theory 

In some of the exploratory test scripts, my goal was to return the predictions with the highest probability of occurring. The model would return those games in which it was 99% confident that the prediction was correct. Of the 250 games in a season's testing set, the model returned only 5-10 games per season in which it was very confident of its choice. These games were usually all correct, but this does not mean the project is strong enough to use for betting on more than a handful of games. 

While the predictions made by the models are around a 60-65% accuracy score for the years tested, win/loss bets do not guarantee that a gambler is profitable, even if above a 50% win rate. This is due to small payouts for wins from favorited teams. A single missed bet can wipe the accumulated winnings gained from 3 or more correct picks.  

A better approach to picking bets is the Kelly Criterion, in which bet sizing and picks are determined on the basis of expected value. More on the Kelly Criterion: https://towardsdatascience.com/betting-optimally-29f283d96669


# Future Fixes
-Cross Validation Train/Test
-Target class imbalance. Maybe upsample/downsample 
-Web scraping for bench rosters
-Injury reports API https://www.fantasybasketballnerd.com/fantasy-basketball-api#injuries
-Machine learning pipeline 
-Tensorflow CNN...





