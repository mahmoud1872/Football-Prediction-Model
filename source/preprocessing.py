import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_teams(df):
    all_teams = list(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    home_encoder = LabelEncoder()
    away_encoder = LabelEncoder()
    home_encoder.fit(all_teams)
    away_encoder.fit(all_teams)

    df["HomeTeam"] = home_encoder.transform(df["HomeTeam"])
    df["AwayTeam"] = away_encoder.transform(df["AwayTeam"])

    hotEncoder = OneHotEncoder(sparse_output=False)

    home_encoded = hotEncoder.fit_transform(df[["HomeTeam"]])
    away_encoded = hotEncoder.fit_transform(df[["AwayTeam"]])

    home_df = pd.DataFrame(home_encoded, columns=[f'home_team_{i}' for i in range(home_encoded.shape[1])])
    away_df = pd.DataFrame(away_encoded, columns=[f'away_team_{i}' for i in range(away_encoded.shape[1])])

    df = pd.concat([df, home_df, away_df], axis=1)
    df = df.drop(columns=["HomeTeam", "AwayTeam"])
    return df

def encode_weekdays(df):
    weekdays = df[["Weekday"]]
    hotEncoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encodedWeekdays = hotEncoder.fit_transform(weekdays)

    weekday_encoded_df = pd.DataFrame(
        encodedWeekdays,
        columns=hotEncoder.get_feature_names_out(['Weekday']),
        index=df.index
    )
    df = pd.concat([df.drop('Weekday', axis=1), weekday_encoded_df], axis=1)
    if 'Weekday_nan' in df.columns:
        df.drop(columns='Weekday_nan', inplace=True)
    return df
