from unittest import TestCase
from data_loader import DataLoader
from data_splitter import DataSplitter

import pandas as pd
import numpy as np


class TestLoadRealData(TestCase):
    def setUp(self):
        feed_dd = DataLoader().load_real_data("../data/real_data_sample.parquet")
        self.feed_df = feed_dd.compute()

    def test_can_load_data(self):
        assert len(self.feed_df['time']) > 0

    def test_time_column_has_correct_type(self):
        assert (type(self.feed_df['time'].iloc[0]) == pd._libs.tslib.Timestamp)

    def test_price_column_has_correct_type(self):
        assert (type(self.feed_df['price'].iloc[0]) == np.float64)

    def test_size_column_has_correct_type(self):
        assert (type(self.feed_df['size'].iloc[0]) == np.float64)

    def test_DELETE(self):
        df = DataLoader().load_real_data("/Users/jamesprince/project-data/2018-03-25.parquet").compute()
        one_hour_and_five = DataSplitter().get_first_n_nanos(df, (60 * 65) * 10 ** 9)
        five_mins = DataSplitter().get_last_n_nanos(one_hour_and_five, (5 * 60) * 10 ** 9)
        five_mins.to_parquet("../data/five-minutes.parquet")
