import configparser
import datetime
import logging
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from backtest_config import BacktestConfig
from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from orderbook.orderbook_creator import reconstruct_orderbook
from orderbook.orderbook_evolutor import OrderBookEvolutor
from stats import Statistics


class TestReport(TestCase):

    def test_get_binom(self):
        n = 100
        p = 0.5

        plt.figure(figsize=(12, 8))
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, n)
        ax.plot(x, binom.pmf(x, n, p), ms=8)
        ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
        plt.show()

    def test_get_binom_coloured(self):
        k = 58
        n = 100
        p = 0.5

        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, k)
        y = np.arange(k, n)
        ax.plot(x, binom.pmf(x, n, p), ms=8)
        ax.plot(y, binom.pmf(y, n, p), ms=8)
        ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
        ax.vlines(y, 0, binom.pmf(y, n, p), colors='r', lw=5, alpha=0.5)
        plt.show()
        plt.figure(figsize=(12, 8))

        plt.xlabel("Correct trials (k)")
        plt.ylabel("Probability")

        plt.savefig("binom.png")

    def test_get_example_cdf(self):
        plt.figure(figsize=(12, 8))
        plt.plot([0, 1, 1, 2, 2, 3, 3], [0, 0, 0.25, 0.25, 0.6, 0.6, 1], 'b', label="CDF")
        plt.plot([0, 2, 2], [0.41, 0.41, 0], 'r', linestyle=':', label="Sample")
        plt.legend()

        plt.show()

    def test_get_orders_per_minute(self):
        product = "LTC-USD"
        root = "/Users/jamesprince/project-data/data/consolidated-feed/"

        st = datetime.datetime(2018, 5, 17, 0, 0, 0)
        et = datetime.datetime(2018, 5, 17, 23, 59, 59)

        feed_df = DataLoader.load_feed(root + product + "/",
                                       st, et, product)

        orders = DataSplitter.get_orders(feed_df)
        limit_orders = DataSplitter.get_limit_orders(orders)

        print(str(len(limit_orders)) + " total limit orders per day for " + product)
        print(str(len(limit_orders) / (24 * 60)) + " limit orders per minute (on average) for " + product)

    def test_orders_per_minute_windowed(self):
        product = "LTC-USD"
        root = "/Users/jamesprince/project-data/data/consolidated-feed/"

        st = datetime.datetime(2018, 5, 17, 0, 0, 0)
        et = datetime.datetime(2018, 5, 17, 23, 59, 59)

        feed_df = DataLoader.load_feed(root + product + "/",
                                       st, et, product)

        orders = DataSplitter.get_orders(feed_df)
        limit_orders = DataSplitter.get_limit_orders(orders)
        market_orders = DataSplitter.get_market_orders(orders)

        trades = DataSplitter.get_trades(feed_df)
        cancels = DataSplitter.get_cancellations(feed_df)

        print("Total limit orders: " + str(len(limit_orders)))
        print("Total market orders: " + str(len(market_orders)))
        print("Total trades: " + str(len(trades)))
        print("Total cancels: " + str(len(cancels)))

        # total_vol = trades['remaining_size'].sum()
        # print("Total traded volume: " + str(total_vol))

        window_minutes = 60
        step_minutes = 5

        times = []
        num_limit_orders = []
        num_market_orders = []
        num_trades = []
        num_cancels = []

        traded_vols = []

        for i in range(0, int((24 * 60) / step_minutes - 1)):
            window_st = st + datetime.timedelta(seconds=i * step_minutes * 60)
            window_et = window_st + datetime.timedelta(seconds=window_minutes * 60)

            limit_orders_this_window = DataSplitter.get_between(limit_orders, window_st, window_et)
            market_orders_this_window = DataSplitter.get_between(market_orders, window_st, window_et)
            trades_this_window = DataSplitter.get_between(trades, window_st, window_et)
            cancels_this_window = DataSplitter.get_between(cancels, window_st, window_et)

            times.append(window_st)
            num_limit_orders.append(len(limit_orders_this_window))
            num_market_orders.append(len(market_orders_this_window))
            num_trades.append(len(trades_this_window))
            num_cancels.append(len(cancels_this_window))

            # vol_this_window = trades_this_window['remaining_size'].sum()
            # traded_vols.append(vol_this_window)

        Statistics.plot_metric_daily_comparison(times, num_limit_orders, num_cancels,
                                                "LTC-USD", st,
                                                step_minutes, window_minutes,
                                                "Limit Orders", "Cancels")

        Statistics.plot_metric_daily(times, num_limit_orders, "LTC-USD", st,
                                     step_minutes, window_minutes, "Limit Orders")
        Statistics.plot_metric_daily(times, num_market_orders, "LTC-USD", st,
                                     step_minutes, window_minutes, "Market Orders")
        Statistics.plot_metric_daily(times, num_trades, "LTC-USD", st,
                                     step_minutes, window_minutes, "Trades")
        Statistics.plot_metric_daily(times, num_cancels, "LTC-USD", st,
                                     step_minutes, window_minutes, "Cancels")
        Statistics.plot_metric_daily(times, traded_vols, "LTC-USD", st,
                                     step_minutes, window_minutes, "Traded Volume")

    def test_real_spread_plot(self):
        plt.figure(figsize=(12, 8))

        product = "LTC-USD"
        root = "/Users/jamesprince/project-data/data/consolidated-feed/"

        st = datetime.datetime(2018, 5, 17, 1, 0, 0)
        et = datetime.datetime(2018, 5, 17, 1, 5, 0)

        feed_df = DataLoader.load_feed(root + product + "/",
                                       st, et, product)

        conf = configparser.ConfigParser()
        conf.read("../config/backtest.ini")
        config = BacktestConfig(conf)

        ob_seq, ob_state = reconstruct_orderbook(config, st, logging.getLogger("test"))

        orderbook_evo = OrderBookEvolutor(ob_state, st, ob_seq)
        res_df = orderbook_evo.evolve_orderbook(feed_df)

        res_df['seconds'] = (res_df['time'] - res_df['time'].iloc[0]).apply(lambda x: x.total_seconds())

        print(res_df)

        limit_orders = DataSplitter.get_limit_orders_from_feed(feed_df)
        limit_orders['seconds'] = (limit_orders['time'] - limit_orders['time'].iloc[0]).apply(
            lambda x: x.total_seconds())

        buy_limit_orders = DataSplitter.get_side("buy", limit_orders)
        sell_limit_orders = DataSplitter.get_side("sell", limit_orders)

        cancels = DataSplitter.get_cancellations(feed_df)

        # print(cancels)

        cancels_merged = cancels.merge(limit_orders, on='order_id', how='left')

        # print(cancels_merged)

        cancels_merged['price'] = cancels_merged['price_x']
        cancels_merged['side'] = cancels_merged['side_x']
        cancels_merged['seconds'] = (cancels_merged['time_x'] - cancels_merged['time_x'].iloc[0]).apply(
            lambda x: x.total_seconds())

        cancels_merged['lifetime'] = abs(cancels_merged['time_x'] - cancels_merged['time_y']).dropna()

        print(cancels_merged)
        median_idx = int(len(cancels_merged['lifetime']) / 2)
        print(cancels_merged['lifetime'].sort_values().iloc[median_idx])

        buy_cancels = DataSplitter.get_side("buy", cancels_merged)
        sell_cancels = DataSplitter.get_side("sell", cancels_merged)

        plt.plot(buy_limit_orders['seconds'], buy_limit_orders['price'], 'r+', label="Buy limit orders")
        plt.plot(sell_limit_orders['seconds'], sell_limit_orders['price'], 'b+', label="Sell limit orders")

        # plt.plot(buy_cancels['seconds'], buy_cancels['price'], 'r+', label="Buy side cancels")
        # plt.plot(sell_cancels['seconds'], sell_cancels['price'], 'b+', label="Sell side cancels")

        plt.plot(res_df['seconds'], res_df['best_bid'], label='Best bid price')
        plt.plot(res_df['seconds'], res_df['best_ask'], label='Best ask price')

        start_price = res_df['midprice'].iloc[0]
        plt.ylim(start_price - 5, start_price + 5)

        plt.legend()
        plt.show()

    def test_sim_spread_plot(self):
        plt.figure(figsize=(12, 8))

        product = "LTC-USD"
        root = "/Users/jamesprince/project-data/results/sims/LTC-USD/2018-05-17/01:00:00/"

        st = datetime.datetime(2018, 5, 17, 1, 0, 0)
        et = datetime.datetime(2018, 5, 17, 1, 5, 0)

        all_sims = DataLoader().load_sim_data(root)
        # orders_dd, trades_dd, cancels_dd, midprices_dd, best_bids_dd, best_asks_dd

        orders_df = all_sims[0][0].compute()
        cancels_df = all_sims[0][2].compute()
        midprice_df = all_sims[0][3].compute()

        conf = configparser.ConfigParser()
        conf.read("../config/backtest.ini")
        config = BacktestConfig(conf)

        # limit_orders = DataSplitter.get_limit_orders(orders_df)
        # limit_orders['seconds'] = (limit_orders['time'] - limit_orders['time'].iloc[0]).apply(
        #     lambda x: x.total_seconds())
        #
        # buy_limit_orders = DataSplitter.get_side("buy", limit_orders)
        # sell_limit_orders = DataSplitter.get_side("sell", limit_orders)
        #
        # plt.plot(buy_limit_orders['seconds'], buy_limit_orders['price'], 'r+', label="Buy limit orders")
        # plt.plot(sell_limit_orders['seconds'], sell_limit_orders['price'], 'b+', label="Sell limit orders")

        cancels_df['seconds'] = (cancels_df['time'] - cancels_df['time'].iloc[0]).apply(
            lambda x: x.total_seconds())

        buy_cancels = DataSplitter.get_side("buy", cancels_df)
        sell_cancels = DataSplitter.get_side("sell", cancels_df)

        plt.plot(buy_cancels['seconds'], buy_cancels['price'], 'r+', label="Buy side cancels")
        plt.plot(sell_cancels['seconds'], sell_cancels['price'], 'b+', label="Sell side cancels")

        # plt.plot(res_df['seconds'], res_df['best_bid'], label='Best bid price')
        # plt.plot(res_df['seconds'], res_df['best_ask'], label='Best ask price')

        start_price = midprice_df['price'].iloc[0]
        plt.ylim(start_price - 5, start_price + 5)

        plt.legend()
        plt.show()
