import pandas as pd

def parser(x):
    return pd.to_datetime(x)

series = pd.read_csv('new_data/gilman1_1_mod.csv', header=0, parse_dates=[0],index_col = 0, squeeze = True)
