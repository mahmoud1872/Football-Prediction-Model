# Football-Prediction-Model
Machine learning project predicting English Premier League match outcomes using historical data and betting odds.


### Premier League Match Outcome Prediction

  This project is designed to predict the outcome of English Premeir League (EPL) matches (Home Win, Draw, Away Win) using historical match data and machine learning    models.

  data collected from [football-data.co.uk](https://www.football-data.co.uk/englandm.php)

### Project Overview

  Combined 11 seasons of EPL match data into a single dataset (csv).

  Performed feature engineering such as rolling averages of goals scored/conceded, bookmaker odds normalization, and time-based features (weekday, month, year).

  Applied some data preprocessing such as one-hot encoding for categorical features and normalization for numerical features (odds data).

Tested several machine learning models:

  • Logistic Regression

  • Support Vector Classifier (SVC)

  • Random Forest

  • Gradient Boosting (best performing)

  • Evaluated models using accuracy, precision, recall, F1-score, and confusion matrix.

### Exploring data for analysis

  Distribution of match outcomes.

  Betting Odds Distribution

  Correlation analysis between features.

  Relationship between odds and actual match results.

### Used

  • Python

  • Pandas, NumPy for data processing

  • Matplotlib for visualizations

  • Scikit-learn for preprocessing and modeling

  • Gradient Boosting Classifier for prediction

### Results

  The Gradient Boosting model achieved the best performance with about 55% accuracy , which might seem low for a ML model, but due to the large number of   unpredictable factors such as the dataset being highly imbalanced (draws are much harder to predict) and luck, it is considered good accuracy.
Very strong models which contain complex features and live data rarely achieve 60–65%.
