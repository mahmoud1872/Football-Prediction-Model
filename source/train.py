from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from loadData import load_and_merge
from addingFeatures import add_rolling_features, add_date_features, normalize_odds
from preprocessing import encode_teams, encode_weekdays
from evalutaeModel import plot_confusion_matrix
df = load_and_merge("../Datasets/Raw_data/*.csv")

df = add_rolling_features(df)
df = add_date_features(df)
df = normalize_odds(df)

df = df[['HomeTeam','AwayTeam','BbAvH','BbAvD','BbAvA','FTR',
         'AwayTeamGoals', 'HomeTeamConceded', 'AwayTeamConceded', 'HomeTeamGoals','Weekday']]

df.to_csv("../Datasets/PL11.csv")

df = encode_teams(df)
df = encode_weekdays(df)

X = df.drop(columns="FTR")
y = df["FTR"].map({'H':0,'D':1,'A':2})

X.dropna(inplace=True)
y = y[X.index]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.005, 0.01, 0.05],
    "max_depth": [2,3,4],
    "subsample": [0.8,1.0],
    "min_samples_split": [2,5,10]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(x_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_model = grid.best_estimator_
best_model.fit(x_train, y_train)


model = GradientBoostingClassifier(learning_rate= 0.005, max_depth= 2, min_samples_split= 2, n_estimators= 400, subsample= 0.8)
model.fit(x_train , y_train)


y_pred = model.predict(x_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plot_confusion_matrix(y_test , y_pred)