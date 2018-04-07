import dask.dataframe as dd
import pandas as pd
import numpy as np
import fitting
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import stats


def graph_time_delta(orders_df: dd):
    order_time_delta_df = pd.Series(orders_df['time'].apply(lambda x: date_to_unix(x, 'ns') / 1e6).diff())
    get_distribution(order_time_delta_df, 'BTC-USD inter-order arrival times', 'inter order time (ms)', bins=100)


def graph_order_sizes(orders_df: dd):
    size_df = pd.Series(orders_df['size'].astype('float64').tolist())
    get_distribution(size_df, 'BTC-USD Order size', 'Order Size', 200)


def graph_trade_sizes(trades_df: dd):
    order_id_map = trades_df[['order_id', 'size']].drop_duplicates()
    traded_order_id_set = pd.DataFrame(trades_df['order_id'].unique(), columns=['order_id'])
    joined = order_id_map.join(traded_order_id_set.set_index('order_id'), on='order_id', how='inner')
    result = pd.Series(joined.dropna(axis=0, how='any').reset_index(drop=True)['size'].astype('float64').tolist())
    get_distribution(result, 'BTC-USD Trade Order Size', 'Trade Order Size')


def graph_trade_price(trades_df: dd):
    #
    y = trades_df['price'].astype('float64').fillna(method='ffill')
    print(y)
    x = trades_df['time'].astype('datetime64[ns]').apply(lambda x: date_to_unix(x, 'ns') / 1e6)
    print(x)

    plt.figure(figsize=(12, 8))


    # import matplotlib.ticker as plticker
    #
    # loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    # plt.xaxis.set_major_locator(loc)
    # plt.xticks()

    plt.plot(x, y, 'r+')
    # plt.yticks(np.arange(y.min(), y.max(), step=100))

    plt.xlabel('Time (ns since epoch)')
    plt.ylabel('Price ($)')

    # ax.set_title('BTC-USD trade price')
    # ax.set_xlabel('time')


def graph_sides(df: dd, product: str) -> None:
    btc_usd_price_buy = pd.Series(stats.get_side('buy', df)['price'].astype('float64').tolist())
    btc_usd_price_sell = pd.Series(stats.get_side('sell', df)['price'].astype('float64').tolist())

    get_distribution(btc_usd_price_buy, product + ' buy side', 'Price ($)', bins=50)
    get_distribution(btc_usd_price_sell, product + ' sell side', 'Price ($)', bins=50)


def date_to_unix(s, unit: str):
    return pd.to_datetime(s, unit=unit).value


def get_distribution(data: pd.Series, description: str, xlabel: str, bins=20, std_devs: int=2):
    sample_size = 10000

    data = stats.keep_n_std_dev(data, std_devs)
    if len(data) > sample_size:
        data = data.sample(n=sample_size)
    data = stats.keep_n_std_dev(data, std_devs)

    print(data)

    fitting.best_fit_with_graphs(data, description, xlabel, bins=bins)