from typing import Dict, Union, Any

import dask.dataframe as dd
import pandas as pd

from data.data_splitter import DataSplitter


class Statistics:

    def __init__(self):
        """
        A class which generates a bunch of statistics related to trade feed data
        """

    def modifications(self, df: dd):
        pass

    @staticmethod
    def get_num_reason(reason: str, df: dd):
        num = len(df[df['reason'] == reason])
        return num

    @staticmethod
    def get_num_type(t: str, df: dd) -> int:
        num = len(df[df['type'] == t])
        return num

    @staticmethod
    def get_mean(col_name: str, df: dd) -> dd:
        return df[col_name].astype('float64').mean()

    @staticmethod
    def get_std_dev(col_name: str, df: dd) -> dd:
        return df[col_name].astype('float64').std()

    @staticmethod
    def get_ratio(a: float, b: float):
        total = a+b

        if total == 0:
            return 0, 0

        a_ratio = a / total
        b_ratio = b / total

        return a_ratio, b_ratio

    @staticmethod
    def get_buy_sell_order_ratio(df: dd) -> (float, float):
        num_buys = len(DataSplitter.get_side("buy", df))
        num_sells = len(DataSplitter.get_side("sell", df))

        return Statistics.get_ratio(num_buys, num_sells)

    @staticmethod
    def get_buy_sell_volume_ratio(df: dd):
        buys = DataSplitter.get_side("buy", df)
        sells = DataSplitter.get_side("sell", df)

        buy_vol = buys['size'].sum()
        sell_vol = sells['size'].sum()

        return Statistics.get_ratio(buy_vol, sell_vol)

    @staticmethod
    def get_limit_market_order_ratio(df: dd):
        limits = DataSplitter.get_limit_orders(df)
        markets = DataSplitter.get_market_orders(df)

        num_limits = len(limits)
        num_markets = len(markets)

        return Statistics.get_ratio(num_limits, num_markets)

    # PRE: df must be trades only
    @staticmethod
    def get_price_over_time(trades_df: dd) -> pd.DataFrame:
        price_times_df = trades_df[['time', 'price']].dropna()
        price_times_df.rename(index=str, columns={"price": "most_recent_trade_price"}, inplace=True)
        price_times_df['most_recent_trade_price'] = price_times_df['most_recent_trade_price'].astype('float64')
        return price_times_df.drop_duplicates()

    @staticmethod
    def get_feed_stats(df: dd) -> Dict[str, Union[int, Any]]:
        """Calculate and print some statistics relating to the data feed"""
        stats = {'num_total_msgs': get_total(df), 'num_trades': Statistics.get_num_reason('filled', df),
                 'num_cancel': Statistics.get_num_reason('canceled', df), 'num_received': Statistics.get_num_type('received', df),
                 'num_open': Statistics.get_num_type('open', df), 'num_done': Statistics.get_num_type('done', df),
                 'num_match': Statistics.get_num_type('match', df), 'num_change': Statistics.get_num_type('change', df),
                 'avg_trade_price': Statistics.get_mean('price', DataSplitter.get_trades(df)),
                 'std_dev_trade_price': Statistics.get_std_dev('price', DataSplitter.get_trades(df))}

        return stats

    @staticmethod
    def get_order_stats(df: dd) -> Dict[Union[str, Any], Union[float, Any]]:
        stats = {'buy_order_ratio': Statistics.get_buy_sell_order_ratio(df)[0],
                 'sell_order_ratio': Statistics.get_buy_sell_order_ratio(df)[1],
                 'buy_volume_ratio': Statistics.get_buy_sell_volume_ratio(df)[0],
                 'sell_volume_ratio': Statistics.get_buy_sell_volume_ratio(df)[1],
                 'avg_order_size': Statistics.get_mean('size', df), 'std_dev_order_size': Statistics.get_std_dev('size', df),
                 'avg_sell_order_size': Statistics.get_mean('size', DataSplitter.get_side('sell', df)),
                 'std_dev_sell_order_size': Statistics.get_std_dev('size', DataSplitter.get_side('sell', df)),
                 'avg_buy_order_size': Statistics.get_mean('size', DataSplitter.get_side('buy', df)),
                 'std_dev_buy_order_size': Statistics.get_std_dev('size', DataSplitter.get_side('buy', df)),
                 'avg_price': df['price'].astype('float64').mean(),
                 'std_dev_price': df['price'].astype('float64').std(),
                 'avg_sell_order_price': Statistics.get_mean('price', DataSplitter.get_side('sell', df)),
                 'std_dev_sell_price': Statistics.get_std_dev('price', DataSplitter.get_side('sell', df)),
                 'avg_buy_price': Statistics.get_mean('price', DataSplitter.get_side('buy', df)),
                 'std_dev_buy_order_price': Statistics.get_std_dev('price', DataSplitter.get_side('buy', df))}

        return stats


def get_total(df: dd) -> int:
    total = len(df.values)
    print("number of rows: " + str(total))
    return len(df.values)
