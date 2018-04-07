import dask.dataframe as dd
import pandas as pd
import numpy as np
import fitting
import argparse
import matplotlib
import matplotlib.pyplot as plt
import qqplot
import stats

parser = argparse.ArgumentParser(description='Consolidate multiple parquet files into just one.')
parser.add_argument('input_file', metavar='-i', type=str, nargs=1,
                    help='input file for which you want some info/statistics')

args = parser.parse_args()

# file = "/Users/jamesprince/Google Drive/Imperial/4/Project/data/2018-03-25.parquet"
file = args.input_file


def total(df: dd) -> int:
    total = len(df.values)
    print("number of rows: " + str(total))
    return len(df.values)


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    print("number of sell side interactions: " + str(num_sells))
    print("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells





# num_total = total(df)
# num_btc_usd = total(btc_usd_df)
# print("percentage of orders of this market vs whole feed: " + str((100*num_btc_usd) / num_total) + "%")
#


input_df = dd.read_parquet(file).compute()
# print(df['product_id'].unique())
# print(df)
btc_usd_df = input_df[input_df['product_id'] == 'BTC-USD']

# calculate_stats(btc_usd_df)

# graph_sides(btc_usd_df, "BTC-USD")
btc_usd_price_buy = pd.Series(stats.get_side('buy', input_df)['price'].astype('float64').tolist())

sample_size = 100
std_devs = 3

data = keep_n_std_dev(btc_usd_price_buy, std_devs)
if len(btc_usd_price_buy) > sample_size:
    data = btc_usd_price_buy.sample(n=sample_size)
# btc_usd_price_buy = keep_n_std_dev(btc_usd_price_buy, std_devs)
print("here")

theoretical, _ = fitting.best_fit_distribution(data, 200)

print(theoretical)

qqplot.plot(btc_usd_price_buy, theoretical)

plt.show()
