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


def get_distribution(data: pd.Series, description: str, xlabel: str, bins=200, std_devs: int=3):
    sample_size = 10000

    data = keep_n_std_dev(data, std_devs)
    if len(data) > sample_size:
        data = data.sample(n=sample_size)
    data = keep_n_std_dev(data, std_devs)

    fitting.best_fit_with_graphs(data, description, xlabel, bins=bins)


def keep_n_std_dev(data: pd.Series, n: int) -> pd.Series:
    return data[~((data - data.mean()).abs() > n * data.std())]


def graph_sides(df: dd, product: str) -> None:
    btc_usd_price_buy = pd.Series(get_side('buy', df)['price'].astype('float64').tolist())
    btc_usd_price_sell = pd.Series(get_side('sell', df)['price'].astype('float64').tolist())

    get_distribution(btc_usd_price_buy, product + ' buy side', 'Price ($)')
    get_distribution(btc_usd_price_sell, product + ' sell side', 'Price ($)')


def date_to_unix(s, unit: str):
    return pd.to_datetime(s, unit=unit).value


def graph_time_delta(orders_df: dd):
    orders_df = get_orders(btc_usd_df)
    order_time_delta_df = pd.Series(orders_df['time'].apply(lambda x: date_to_unix(x, 'ns') / 1e6).diff())
    get_distribution(order_time_delta_df, 'BTC-USD inter-order arrival times', 'inter order time (ms)', bins=100)


def graph_order_sizes(orders_df: dd):
    size_df = pd.Series(orders_df['size'].astype('float64').tolist())
    get_distribution(size_df, 'BTC-USD Order size', 'Order Size', 200)


def graph_trade_sizes(trades_df: dd):
    order_id_map = trades_df[['order_id', 'size']].drop_duplicates()
    traded_order_id_set = pd.DataFrame(input_df['order_id'].unique(), columns=['order_id'])
    joined = order_id_map.join(traded_order_id_set.set_index('order_id'), on='order_id', how='inner')
    result = pd.Series(joined.dropna(axis=0, how='any').reset_index(drop=True)['size'].astype('float64').tolist())
    get_distribution(result, 'BTC-USD Trade Order Size', 'Trade Order Size')


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
