import dask.dataframe as dd
import pandas as pd


def get_total(df: dd) -> int:
    total = len(df.values)
    print("number of rows: " + str(total))
    return len(df.values)


def keep_n_std_dev(data: pd.Series, n: int) -> pd.Series:
    return data[~((data - data.mean()).abs() > n * data.std())]


def get_side(side: str, df: dd) -> dd:
    return df[df['side'] == side]


def get_trades(df: dd) -> dd:
    return df[df['reason'] == 'filled']


def get_orders(df: dd) -> dd:
    return df[df['type'] == 'received']


def get_cancellations(df: dd) -> int:
    return df[df['reason'] == 'canceled']


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


def get_buy_sell_ratio(df: dd) -> (float, float):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])

    buy_ratio = (100*num_buys) / (num_buys + num_sells)
    sell_ratio = (100 * num_sells) / (num_buys + num_sells)

    return buy_ratio, sell_ratio


def calculate_stats(df: dd) -> None:
    """Calculate and print some statistics based on the """
    num_total_msgs = get_total(df)
    num_trades = get_num_reason('filled', df)
    num_cancel = get_num_reason('canceled', df)

    num_received = get_num_type('received', df)
    num_open = get_num_type('open', df)
    num_done = get_num_type('done', df)
    num_match = get_num_type('match', df)
    num_change = get_num_type('change', df)
    # sides(df)

    buy_ratio, sell_ratio = get_buy_sell_ratio(df)
    print("Ratio (buy/sell): " + str(buy_ratio) + ":" + str(sell_ratio))

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
