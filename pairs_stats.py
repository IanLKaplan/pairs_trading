from datetime import datetime
from typing import Tuple
import matplotlib.pyplot as plt

import pandas as pd
from tabulate import tabulate

from utils.find_date_index import findDateIndex
from read_market_data.MarketData import MarketData


trading_days = 252
s_and_p_data = 's_and_p_data'
start_date_str = '2007-01-03'
start_date: datetime = datetime.fromisoformat(start_date_str)
d2008_date_str = '2008-01-03'
d2008_start_date: datetime = datetime.fromisoformat(d2008_date_str)

apple_tuple: Tuple = ('AAPL', 'MPWR')
market_data = MarketData(start_date=start_date, path=s_and_p_data)

close_prices_df = market_data.get_close_data(list(apple_tuple))
close_index = close_prices_df.index

half_year = int(trading_days/2)


def normalize_df(data_df: pd.DataFrame) -> pd.DataFrame:
    min_s = data_df.min()
    max_s = data_df.max()
    norm_df = (data_df - min_s) / (max_s - min_s)
    return norm_df


def add_mean(data_df: pd.DataFrame) -> pd.DataFrame:
    mean_df = pd.DataFrame((data_df[data_df.columns[0]] + data_df[data_df.columns[1]]) / 2)
    mean_df.columns = ['mean']
    data_mean_df = pd.concat([data_df, mean_df], axis=1)
    return data_mean_df


def add_stats(data_df: pd.DataFrame) -> pd.DataFrame:
    mean_df = pd.DataFrame((data_df[data_df.columns[0]] + data_df[data_df.columns[1]]) / 2)
    mean_df.columns = ['mean']
    data_mean_df = add_mean(data_df)
    mean_s = data_mean_df['mean']
    stddev_s = mean_s.std()
    stddev_high = pd.DataFrame(mean_s + stddev_s)
    stddev_low = pd.DataFrame(mean_s - stddev_s)
    stddev_high.columns = ['StdDev high']
    stddev_low.columns = ['StdDev low']
    data_stats_df = pd.concat([data_mean_df, stddev_high, stddev_low ], axis=1)
    return data_stats_df


d2007_ix = 0
d2008_ix = findDateIndex(close_index, d2008_start_date)

d2007_close_df = close_prices_df.iloc[d2007_ix:d2007_ix+half_year]
d2008_close_df = close_prices_df.iloc[d2008_ix:d2008_ix+half_year]

d2007_cor = round(d2007_close_df.corr().iloc[0,1], 2)
d2008_cor = round(d2008_close_df.corr().iloc[0,1], 2)
cor_df = pd.DataFrame([d2007_cor, d2008_cor])
cor_df.index = ['2007', '2008']
cor_df.columns = [f'correlation [{close_prices_df.columns[0]},{close_prices_df.columns[1]}]']

print()
print(tabulate(cor_df, headers=[*cor_df.columns], tablefmt='fancy_grid'))

scale = 0.75

d2007_norm_df = normalize_df(d2007_close_df)
d2007_stats_df = add_stats(d2007_norm_df)
d2008_norm_df = normalize_df(d2008_close_df)
d2008_stats_df = add_stats(d2008_norm_df)

d2007_stats_df.plot(grid=True, title=f'normalized AAPL/MPWR and mean Jan 3, 2007', figsize=(10, 6))
plt.show()
d2008_stats_df.plot(grid=True, title=f'normalized AAPL/MPWR and mean Jan 3, 2008', figsize=(10, 6))
plt.show()

pass
