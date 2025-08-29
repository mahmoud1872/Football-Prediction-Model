import pandas as pd
import numpy as np

def add_rolling_features(df):
    df['HomeTeamGoals'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df['AwayTeamGoals'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df['HomeTeamConceded'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df['AwayTeamConceded'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())

    for col in ["HomeTeamGoals","AwayTeamGoals","HomeTeamConceded","AwayTeamConceded"]:
        df[col] = df[col].fillna(df[col].mean()).round(2)
    return df

def add_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Weekday'] = df['Date'].dt.weekday
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df = df.drop(columns=['Date'])
    
    month_radians = 2 * np.pi * (df["Month"] - 1) / 12
    df['month_sin'] = np.sin(month_radians)
    df['month_cos'] = np.cos(month_radians)
    df = df.drop(columns=["Month"])
    return df

def normalize_odds(df):
    df["Prob_H"] = 1 / df["BbAvH"]
    df["Prob_D"] = 1 / df["BbAvD"]
    df["Prob_A"] = 1 / df["BbAvA"]
    total = df["Prob_H"] + df["Prob_D"] + df["Prob_A"]
    df["Prob_H"] /= total
    df["Prob_D"] /= total
    df["Prob_A"] /= total
    return df
