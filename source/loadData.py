import pandas as pd
import glob

def load_and_merge(path="../Datasets/Raw_data/*.csv"):
    files = glob.glob(path)
    df_list = [pd.read_csv(f) for f in files]

    for file in df_list:
        file.rename(columns={
            'MaxH': 'BbMxH', 'MaxD': 'BbMxD', 'MaxA': 'BbMxA',
            'AvgH': 'BbAvH', 'AvgD': 'BbAvD', 'AvgA': 'BbAvA'
        }, inplace=True)

    df = pd.concat(df_list, ignore_index=True)
    return df
