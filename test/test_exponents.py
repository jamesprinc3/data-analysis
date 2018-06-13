import configparser
import datetime
import logging
from unittest import TestCase

import nolds
import numpy as np

from backtest_config import BacktestConfig
from data.data_loader import DataLoader
from orderbook.orderbook_creator import reconstruct_orderbook
from orderbook.orderbook_evolutor import OrderBookEvolutor
from stats import Statistics


class TestExponent(TestCase):

    def setUp(self):
        self.root = "/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/"

    def test_lypaunov(self):
        st = datetime.datetime(2018, 5, 17, 1, 0, 0)
        et = datetime.datetime(2018, 5, 17, 1, 30, 0)

        conf = configparser.ConfigParser()
        conf.read("../config/backtest.ini")
        config = BacktestConfig(conf)

        ob_seq, ob_state = reconstruct_orderbook(config, st, logging.getLogger("test"))

        orderbook_evo = OrderBookEvolutor(ob_state, st, ob_seq)

        feed_df = DataLoader.load_feed(self.root, st, et, "LTC-USD")
        evo = orderbook_evo.evolve_orderbook_discrete(feed_df, 1)

        prices = np.asarray(evo['midprice'].dropna(), dtype=np.float32)

        print(prices)

        # print(len(prices))

        res = nolds.lyap_e(prices)

        print(res)

    def test_lypaunov_windowed(self):
        st = datetime.datetime(2018, 5, 17, 1, 0, 0)
        et = datetime.datetime(2018, 5, 17, 23, 0, 0)

        conf = configparser.ConfigParser()
        conf.read("../config/backtest.ini")
        config = BacktestConfig(conf)

        ob_seq, ob_state = reconstruct_orderbook(config, st, logging.getLogger("test"))

        orderbook_evo = OrderBookEvolutor(ob_state, st, ob_seq)

        feed_df = DataLoader.load_feed(self.root, st, et, "LTC-USD")
        evo = orderbook_evo.evolve_orderbook_discrete(feed_df, 1)

        window_minutes = 30
        step_minutes = 5
        num_samples = int((et - st).total_seconds() / (step_minutes * 60))

        times = []
        lyap_exps = []

        for i in range(0, num_samples):
            window_st = st + datetime.timedelta(seconds=i * step_minutes * 60)
            window_et = window_st + datetime.timedelta(seconds=window_minutes * 60)

            evo_filt = evo[evo['time'] > window_st]
            evo_filt = evo_filt[evo_filt['time'] < window_et]
            midprices = evo_filt['midprice'].dropna()

            prices = np.asarray(midprices, dtype=np.float32)
            print(prices)

            res = nolds.lyap_e(prices)
            print(res)

            times.append(window_st)
            lyap_exps.append(res[0])

        Statistics.plot_lyapunov_exponent(times, lyap_exps, "LTC-USD", st, step_minutes, window_minutes)

    def test_hurst(self):
        st = datetime.datetime(2018, 5, 17, 1, 0, 0)
        et = datetime.datetime(2018, 5, 17, 2, 0, 0)
        _, trades, _ = DataLoader.load_split_data("/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/", st,
                                                  et, "LTC-USD")

        prices = np.asarray(trades['price'].dropna(), dtype=np.float32)

        print(prices)

        # print(len(prices))

        # res = nolds.hurst_rs(prices)
        res = nolds.dfa(prices)

        print(res)

    def test_hurst_windowed(self):
        day = 17
        product = "LTC-USD"
        for i in range(0, 1):
            # day += 1
            month = 5
            st = datetime.datetime(2018, month, day, 0, 0, 0)
            et = datetime.datetime(2018, month, day, 23, 59, 59)
            _, trades, _ = DataLoader.load_split_data("/Users/jamesprince/project-data/data/consolidated-feed/"
                                                      + product + "/",
                                                      st,
                                                      et, product)

            window_minutes = 60
            step_minutes = 10
            times, hurst_exps = Statistics.get_hurst_exponent_over_time(trades, st, et, step_minutes, window_minutes)
            Statistics.plot_metric_daily(times, hurst_exps, product, st, step_minutes, window_minutes, "Hurst Exponent")
