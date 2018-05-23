from unittest import TestCase
from data.data_splitter import DataSplitter

import pandas as pd
import datetime


class TestGetFirstNNanos(TestCase):

    def setUp(self):
        epoch = datetime.datetime.utcfromtimestamp(0)
        time_beyond_end = datetime.datetime.utcfromtimestamp(10)
        times = [epoch, time_beyond_end]
        # logger.debug(times)
        self.df = pd.DataFrame({"time": times})
        self.df['time'] = self.df['time'].astype('datetime64[ns]')

    def test_filter_when_cutoff_before_end(self):

        assert len(DataSplitter.get_first_n_nanos(self.df, 5)['time']) == 1

    def test_filter_when_cutoff_after_end(self):
        assert len(DataSplitter.get_first_n_nanos(self.df, 15 * 10**9)['time']) == 2
