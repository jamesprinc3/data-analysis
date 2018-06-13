import datetime
from typing import Dict, Union, Any

import dask.dataframe as dd
import nolds
import numpy as np
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
    def get_reason_count(reason: str, df: dd):
        num = len(df[df['reason'] == reason])
        return num

    @staticmethod
    def get_type_count(t: str, df: dd) -> int:
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
    def format_ratio(a, b, dp=2):
        format_str = "%." + str(dp) + "f"

        return format_str % a, format_str % b

    @staticmethod
    def get_buy_sell_ratio_str(df: dd, dp=2) -> (float, float):
        num_buys, num_sells = Statistics.get_buy_sell_ratio(df)
        return Statistics.format_ratio(num_buys, num_sells, dp)

    @staticmethod
    def get_buy_sell_ratio(df: dd) -> (float, float):
        num_buys = len(DataSplitter.get_side("buy", df))
        num_sells = len(DataSplitter.get_side("sell", df))

        return Statistics.get_ratio(num_buys, num_sells)

    @staticmethod
    def get_buy_sell_vol_ratio_str(df: dd, dp=2):
        buy_vol, sell_vol = Statistics.get_buy_sell_volume_ratio(df)
        return Statistics.format_ratio(buy_vol, sell_vol, dp)

    @staticmethod
    def get_buy_sell_volume_ratio(df: dd):
        buys = DataSplitter.get_side("buy", df)
        sells = DataSplitter.get_side("sell", df)

        buy_vol = buys['size'].sum()
        sell_vol = sells['size'].sum()

        return Statistics.get_ratio(buy_vol, sell_vol)

    @staticmethod
    def get_limit_market_order_ratio(df: dd):
        limits = DataSplitter.get_limit_orders_from_feed(df)
        markets = DataSplitter.get_market_orders_from_feed(df)

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
        stats = {'num_total_msgs': get_total(df), 'num_trades': Statistics.get_reason_count('filled', df),
                 'num_cancel': Statistics.get_reason_count('canceled', df),
                 'num_received': Statistics.get_type_count('received', df),
                 'num_open': Statistics.get_type_count('open', df), 'num_done': Statistics.get_type_count('done', df),
                 'num_match': Statistics.get_type_count('match', df),
                 'num_change': Statistics.get_type_count('change', df),
                 'avg_trade_price': Statistics.get_mean('price', DataSplitter.get_trades(df)),
                 'std_dev_trade_price': Statistics.get_std_dev('price', DataSplitter.get_trades(df))}

        return stats

    @staticmethod
    def get_order_stats(df: dd) -> Dict[Union[str, Any], Union[float, Any]]:
        stats = {'buy_order_ratio': Statistics.get_buy_sell_ratio(df)[0],
                 'sell_order_ratio': Statistics.get_buy_sell_ratio(df)[1],
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

    @staticmethod
    def get_hurst_exponent_over_time(trades, st, et, step_minutes, window_minutes):
        num_steps = ((et - st).total_seconds() / 60) / step_minutes
        hurst_exps = []
        times = []
        for i in range(0, int(num_steps)):
            iter_st = st + datetime.timedelta(minutes=step_minutes * i)
            iter_et = iter_st + datetime.timedelta(minutes=window_minutes)

            window = DataSplitter.get_between(trades, iter_st, iter_et)
            prices = np.asarray(window['price'].dropna(), dtype=np.float32)

            if len(prices) == 0:
                continue

            hurst_exp = nolds.hurst_rs(prices)
            # hurst_exp = nolds.dfa(prices) - 1
            print(hurst_exp)
            if 0 < hurst_exp < 1:
                hurst_exps.append(hurst_exp)
                times.append(iter_st)
            else:
                pass
        return times, hurst_exps

    @staticmethod
    def plot_metric_daily(times, vals, product, st, step_minutes, window_minutes, name: str):
        hours = list(map(lambda t: (t - st).seconds / 3600, times))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.ylim(0, 1)
        plt.plot(hours, vals)
        plt.xlabel("Hour of day")
        plt.ylabel(name)
        plt.title(product + " " + st.date().isoformat() + " " + name + " plotted every " + str(
            step_minutes) + " minutes, window size of " + str(window_minutes) + " minutes")
        plt.show()
        # plt.savefig(
        #     "/Users/jamesprince/project-data/results/" + name + "/" + product + "/" + st.date().isoformat() + ".png")

    @staticmethod
    def plot_metric_daily_comparison(times, vals1, vals2, product, st, step_minutes, window_minutes, name1: str,
                                     name2: str):
        hours = list(map(lambda t: (t - st).seconds / 3600, times))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        # plt.ylim(0, 1)
        plt.plot(hours, vals1, label=name1)
        plt.plot(hours, vals2, label=name2)
        plt.xlabel("Hour of day")
        plt.title(product + " " + st.date().isoformat() + " " + name1 + " plotted against " + name2 + " every " + str(
            step_minutes) + " minutes, window size of " + str(window_minutes) + " minutes")

        plt.legend()
        plt.show()

    # plt.savefig(
    #     "/Users/jamesprince/project-data/results/" + name + "/" + product + "/" + st.date().isoformat() + ".png")

    @staticmethod
    def plot_lyapunov_exponent(times, lyap_exps, product, st, step_minutes, window_minutes):
        hours = list(map(lambda t: (t - st).seconds / 3600, times))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.ylim(0, 1)
        plt.plot(hours, lyap_exps)
        plt.xlabel("Hour of day")
        plt.ylabel("Lyapunov Exponent")
        plt.title(st.date().isoformat() + "Lyapunov exponent plotted every " + str(
            step_minutes) + " minutes, window size of " + str(window_minutes) + " minutes")
        plt.savefig(
            "/Users/jamesprince/project-data/results/lyapunov/" + product + "/" + st.date().isoformat() + ".png")
        plt.show()

    @staticmethod
    def get_lyapunov_exponent_over_time(trades, st, et, step_minutes, window_minutes):
        num_steps = ((et - st).total_seconds() / 60) / step_minutes
        lyap_exps = []
        times = []
        for i in range(0, int(num_steps)):
            iter_st = st + datetime.timedelta(minutes=step_minutes * i)
            iter_et = iter_st + datetime.timedelta(minutes=window_minutes)

            window = DataSplitter.get_between(trades, iter_st, iter_et)
            prices = np.asarray(window['price'].dropna(), dtype=np.float32)

            if len(prices) == 0:
                continue

            lyap_exp = nolds.lyap_r(prices)
            if lyap_exp > 0:
                lyap_exps.append(lyap_exp)
                times.append(iter_et)
            else:
                pass
        return times, lyap_exps


def get_total(df: dd) -> int:
    total = len(df.values)
    print("number of rows: " + str(total))
    return len(df.values)
