import datetime
from unittest import TestCase

import nolds
import numpy as np

from data.data_loader import DataLoader
from stats import Statistics


class TestExponent(TestCase):

    def test_lypaunov(self):
        st = datetime.datetime(2018, 5, 17, 0, 0, 0)
        et = datetime.datetime(2018, 5, 17, 2, 0, 59)
        _, trades, _ = DataLoader.load_split_data("/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/", st,
                                                  et, "LTC-USD")

        prices = np.asarray(trades['price'].dropna(), dtype=np.float32)

        print(prices)

        # print(len(prices))

        res = nolds.lyap_e(prices)

        print(res)

    def test_husrt(self):
        st = datetime.datetime(2018, 5, 17, 0, 0, 0)
        et = datetime.datetime(2018, 5, 18, 23, 0, 59)
        _, trades, _ = DataLoader.load_split_data("/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/", st,
                                                  et, "LTC-USD")

        prices = np.asarray(trades['price'].dropna(), dtype=np.float32)

        print(prices)

        # print(len(prices))

        res = nolds.hurst_rs(prices)

        print(res)

    def test_hurst_windowed(self):
        day = 25
        product = "LTC-USD"
        for i in range(0, 1):
            day += 1
            month = 5
            st = datetime.datetime(2018, month, day, 0, 0, 0)
            et = datetime.datetime(2018, month, day, 1, 59, 59)
            _, trades, _ = DataLoader.load_split_data("/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/",
                                                      st,
                                                      et, product)

            window_minutes = 30
            step_minutes = 10
            times, hurst_exps = Statistics.get_hurst_exponent_over_time(trades, st, et, step_minutes, window_minutes)
            Statistics.plot_hurst_exponent(times, hurst_exps, product, st, step_minutes, window_minutes)
