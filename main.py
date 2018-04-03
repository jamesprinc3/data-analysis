import dask.dataframe as dd
import pandas as pd
import numpy as np
import fitting
import argparse
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Consolidate multiple parquet files into just one.')
parser.add_argument('input_file', metavar='-i', type=str, nargs=1,
                    help='input file for which you want some info/statistics')

args = parser.parse_args()

# file = "/Users/jamesprince/Google Drive/Imperial/4/Project/data/2018-03-25.parquet"
file = args.input_file

df = dd.read_parquet(file).compute()
print(df['product_id'].unique())
btc_usd_df = df[df['product_id'] == 'BTC-USD']

# print(btc_usd_df)


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


def get_side(side: str, df: dd) -> dd:
    return df[df['side'] == side]


def trades(df: dd) -> int:
    num_trades = len(df[df['reason'] == 'filled'])
    print("number of trades: " + str(num_trades))
    return num_trades


def cancellations(df: dd) -> int:
    num_cancel = len(df[df['reason'] == 'canceled'])
    print("number of cancellations: " + str(num_cancel))
    return num_cancel


def modifications(df: dd):
    pass


def get_num_reason(reason: str, df: dd):
    num = len(df[df['reason'] == reason])
    return num


def get_num_type(t: str, df: dd) -> int:
    num = len(df[df['type'] == t])
    return num


def get_mean(col_name: str, df: dd) -> dd:
    return df[col_name].astype('float64').mean()


def get_std_dev(col_name: str, df: dd) -> dd:
    return df[col_name].astype('float64').std()


def calculate_stats(df: dd) -> None:
    """Calculate and print some statistics based on the """
    num_total_msgs = total(df)
    num_trades = get_num_reason('filled', df)
    num_cancel = get_num_reason('canceled', df)

    num_received = get_num_type('received', df)
    num_open = get_num_type('open', df)
    num_done = get_num_type('done', df)
    num_match = get_num_type('match', df)
    num_change = get_num_type('change', df)
    # sides(df)

    avg_order_size = get_mean('size', df)
    std_dev_order_size = get_std_dev('size', df)

    avg_sell_order_size = get_mean('size', get_side('sell', df))
    std_dev_sell_order_size = get_std_dev('size', get_side('sell', df))

    avg_buy_order_size = get_mean('size', get_side('buy', df))
    std_dev_buy_order_size = get_std_dev('size', get_side('buy', df))

    avg_price = df['price'].astype('float64').mean()
    std_dev_price = df['price'].astype('float64').std()

    avg_sell_price = get_mean('price', get_side('sell', df))
    std_dev_sell_price = get_std_dev('price', get_side('sell', df))

    avg_buy_price = get_mean('price', get_side('buy', df))
    std_dev_buy_price = get_std_dev('price', get_side('buy', df))

    print("average order size: " + str(avg_order_size))
    print("std. dev. of order size: " + str(std_dev_order_size))

    print("average sell order size: " + str(avg_sell_order_size))
    print("sell order std. dev: " + str(std_dev_sell_order_size))

    print("average buy order size: " + str(avg_buy_order_size))
    print("buy order std. dev: " + str(std_dev_buy_order_size))

    print("average price: " + str(avg_price))
    print("std. dev. of price: " + str(std_dev_price))

    print("average sell price: " + str(avg_sell_price))
    print("std. dev. of sell price: " + str(std_dev_sell_price))

    print("average buy price: " + str(avg_buy_price))
    print("std. dev. of buy price: " + str(std_dev_buy_price))

    print("percentage of orders canceled: " + str((100*num_cancel) / num_received) + "%")
    print("percentage of orders filled: " + str((100*num_trades) / num_received) + "%")

    print("percentage of received messages: " + str((100*num_received) / num_total_msgs) + "%")
    print("percentage of open messages: " + str((100*num_open) / num_total_msgs) + "%")
    print("percentage of done messages: " + str((100*num_done) / num_total_msgs) + "%")
    print("percentage of match messages: " + str((100*num_match) / num_total_msgs) + "%")
    print("percentage of change messages: " + str((100*num_change) / num_total_msgs) + "%")


def get_distribution(data, description: str):
    data = data[~((data - data.mean()).abs() > 3 * data.std())]
    sampled_data = data.sample(n=1000)
    sampled_data = keep_3_std_dev(sampled_data)

    fitting.best_fit_with_graphs(sampled_data, description)


def keep_3_std_dev(data: pd.Series) -> pd.Series:
    return data[~((data - data.mean()).abs() > 3 * data.std())]

# num_total = total(df)
# num_btc_usd = total(btc_usd_df)
# print("percentage of orders of this market vs whole feed: " + str((100*num_btc_usd) / num_total) + "%")

print("Stats for BTC-USD")
calculate_stats(btc_usd_df)


btc_usd_price_buy = pd.Series(get_side('buy', btc_usd_df)['price'].astype('float64').tolist())
btc_usd_price_sell = pd.Series(get_side('sell', btc_usd_df)['price'].astype('float64').tolist())

get_distribution(btc_usd_price_buy, 'BTC-USD buy side')
get_distribution(btc_usd_price_sell, 'BTC-USD sell side')
plt.show()
