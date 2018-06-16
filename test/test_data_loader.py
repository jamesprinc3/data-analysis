import datetime
from unittest import TestCase

import numpy as np
import pandas as pd

from data.data_loader import DataLoader


class TestLoadRealData(TestCase):
    # def setUp(self):
    #     day = 17
    #     month = 5
    #     product = "LTC-USD"
    #
    #     st = datetime.datetime(2018, month, day, 0, 0, 0)
    #     et = datetime.datetime(2018, month, day, 0, 59, 59)
    #     self.smol_feed_df = DataLoader.load_feed("/Users/jamesprince/project-data/data/consolidated-feed/"
    #                                              + product + "/",
    #                                              st,
    #                                              et, product)
    #
    #     print(self.smol_feed_df)

    def test_load_one_day_parquet(self):
        day = 17
        month = 5
        product = "LTC-USD"

        st = datetime.datetime(2018, month, day, 0, 0, 0)
        et = datetime.datetime(2018, month, day, 1, 59, 59)
        feed_df = DataLoader.load_feed("/Users/jamesprince/project-data/data/consolidated-feed/"
                                       + product + "/",
                                       st,
                                       et, product)

        print(feed_df)

    def test_load_one_day_csv(self):
        day = 17
        month = 5
        product = "LTC-USD"

        st = datetime.datetime(2018, month, day, 0, 0, 0)
        et = datetime.datetime(2018, month, day, 23, 59, 59)
        feed_df = DataLoader.load_feed("/Users/jamesprince/project-data/data/consolidated-feed/"
                                       + product + "/",
                                       st,
                                       et,
                                       product,
                                       "csv")

        print(feed_df)

    def test_can_load_data(self):
        assert len(self.smol_feed_df['time']) > 0

    def test_time_column_has_correct_type(self):
        assert (type(self.smol_feed_df['time'].iloc[0]) == pd._libs.tslib.Timestamp)

    def test_price_column_has_correct_type(self):
        assert (type(self.smol_feed_df['price'].iloc[0]) == np.float64)

    def test_size_column_has_correct_type(self):
        assert (type(self.smol_feed_df['size'].iloc[0]) == np.float64)

    def test_DELETE(self):
        df = pd.read_parquet("/Users/jamesprince/project-data/real/2018-03-25.parquet")
        df = DataLoader().format_dd(df)
        for hour in range(0, 3):
            start_time = datetime.datetime(year=2018, month=3, day=25, hour=hour, minute=0, second=0)
            end_time = datetime.datetime(year=2018, month=3, day=25, hour=hour + 1, minute=0, second=0)

            hour_df = df[start_time < df['time']]
            hour_df = hour_df[hour_df['time'] < end_time]

            hour_df.to_parquet("/Users/jamesprince/project-data/real/2018-03-25/" + str(hour) + ".parquet")
