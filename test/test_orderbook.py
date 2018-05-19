import datetime
import pandas as pd
from unittest import TestCase

from data_loader import DataLoader
from orderbook import OrderBook


class TestLoadOrderbook(TestCase):
    # def setUp(self):
        # feed_dd = DataLoader().load_real_data("../data/real_data_sample.parquet")
        # self.feed_df = feed_dd.compute()

    def test_can_load_data(self):
        df = OrderBook().load_orderbook_state("/Users/jamesprince/project-data/gdax/test-orderbook.json")
        print(df)
        assert not df.empty

    def test_can_reconstruct(self):
        st = datetime.datetime(2018, 5, 16, 18, 0, 0)
        et = datetime.datetime(2018, 5, 16, 18, 10, 0)

        all_data = DataLoader.load_split_data("/Users/jamesprince/project-data/new-nimue-backup/feed-consolidated/",
                                              st, et, "BTC-USD")

        ob_state_df = OrderBook().load_orderbook_state("/Users/jamesprince/project-data/new-nimue-backup/orderbook/BTC-USD/2018-05-16T18:05:08.067228.json")

        ob_residuals = OrderBook().orderbook_residual(*all_data)

        ob_final = OrderBook().get_orderbook(ob_residuals, ob_state_df)

        print(ob_final['size'].apply(pd.to_numeric).sum())

    def test_can_locate_files(self):
        time = datetime.datetime(2018, 5, 16, 18, 0, 0)

        OrderBook.locate_closest_ob_state("/Users/jamesprince/project-data/new-nimue-backup/orderbook/BTC-USD/", time)

    def test_get_spread(self):
        ob_state_df = OrderBook().load_orderbook_state("/Users/jamesprince/project-data/new-nimue-backup/orderbook/BTC-USD/2018-05-16T18:05:08.067228.json")
        print(ob_state_df)

        spread = OrderBook.get_spread(ob_state_df)

        print(spread)