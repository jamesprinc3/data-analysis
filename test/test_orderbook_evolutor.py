import datetime
from unittest import TestCase

from data.data_loader import DataLoader
from orderbook.orderbook_creator import OrderBookCreator
from orderbook.orderbook_evolutor import OrderBookEvolutor


class TestOrderBookEvolutor(TestCase):

    def setUp(self):
        self.real_root = "/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/"
        self.st = datetime.datetime(2018, 5, 17, 0, 58, 45)
        self.et = datetime.datetime(2018, 5, 17, 1, 5, 0)
        self.product = "LTC-USD"

    def get_test_orderbook_evolutor(self):
        feed_data = DataLoader.load_split_data(self.real_root,
                                               self.st, self.et, self.product)
        orderbook_root = "/Users/jamesprince/project-data/data/raw-orderbooks/" + self.product + "/"
        # Have to go an hour back because order book file names are UTC+1
        closest_ob_time, closest_ob_file = OrderBookCreator.locate_closest_ob_state(orderbook_root,
                                                                                    self.st + datetime.timedelta(
                                                                                        hours=1))
        self.feed_start_time = closest_ob_time - datetime.timedelta(hours=1)
        # closest_ob_path = orderbook_root + closest_ob_file
        closest_ob_path = orderbook_root + "2018-05-17T01:58:45.628278.json"
        start_seq, ob_state_df = OrderBookCreator().load_orderbook_state(closest_ob_path)
        # ob_start_state_df = OrderBookCreator().get_orderbook(*feed_data, ob_state_df)
        orderbook = OrderBookEvolutor(ob_state_df, self.st, start_seq)

        print("order book seq: " + str(start_seq))

        return orderbook

    def test_init(self):
        orderbook = self.get_test_orderbook_evolutor()

        assert (len(orderbook.bids_max_heap) > 0)
        assert (len(orderbook.asks_min_heap) > 0)

    def test_evolve(self):
        orderbook_evolutor = self.get_test_orderbook_evolutor()

        feed_df = DataLoader().load_feed(self.real_root, self.feed_start_time, self.et, self.product)

        seq_list = feed_df['sequence'].sort_values().values.tolist()

        print("feed start: " + str(seq_list[0]))

        summary_df = orderbook_evolutor.evolve_orderbook_discrete(feed_df, step_seconds=1)

        print(summary_df)
