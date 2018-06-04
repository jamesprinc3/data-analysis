import datetime
from unittest import TestCase

import nolds
import numpy as np

from data.data_loader import DataLoader
from modes.sample import Sample


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

