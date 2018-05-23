import datetime
from unittest import TestCase
from data.data_loader import DataLoader

import pandas as pd
import numpy as np


class TestLoadRealData(TestCase):
    # def setUp(self):
        # feed_dd = DataLoader().load_real_data("../data/real_data_sample.parquet")
        # self.feed_df = feed_dd.compute()

    def test_can_load_data(self):
        assert len(self.feed_df['time']) > 0

    def test_time_column_has_correct_type(self):
        assert (type(self.feed_df['time'].iloc[0]) == pd._libs.tslib.Timestamp)

    def test_price_column_has_correct_type(self):
        assert (type(self.feed_df['price'].iloc[0]) == np.float64)

    def test_size_column_has_correct_type(self):
        assert (type(self.feed_df['size'].iloc[0]) == np.float64)

    def test_DELETE(self):
        df = pd.read_parquet("/Users/jamesprince/project-data/real/2018-03-25.parquet")
        df = DataLoader().format_dd(df)
        for hour in range(0, 3):
            start_time = datetime.datetime(year=2018, month=3, day=25, hour=hour, minute=0, second=0)
            end_time = datetime.datetime(year=2018, month=3, day=25, hour=hour+1, minute=0, second=0)

            hour_df = df[start_time < df['time']]
            hour_df = hour_df[hour_df['time'] < end_time]

            hour_df.to_parquet("/Users/jamesprince/project-data/real/2018-03-25/" + str(hour) + ".parquet")
