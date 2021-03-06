import logging
import os
from datetime import datetime

import dask.dataframe as dd
import pandas as pd
from pandas.errors import EmptyDataError

from data.data_splitter import DataSplitter


class DataLoader:
    """Loads data and formats it appropriately so that we can always assume
    timestamps and floats are not string types to avoid silly conversions all over the place"""
    logger = logging.getLogger("DataLoader")

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def format_midprice_dd(self, df: dd) -> dd:
        df['time'] = df['time'].astype('datetime64[ns]')
        df['price'] = df['price'].astype('float64')

        return df

    def format_dd(self, df: dd) -> dd:
        df['time'] = df['time'].astype('datetime64[ns]')
        df['price'] = df['price'].astype('float64')
        df['size'] = df['size'].astype('float64')

        return df

    def load_sim_data(self, root, start_index=0, end_index=None) -> (dd, dd, dd):
        """
        :param root: directory that contains the output from OrderBookSimulator
        :param start_index: the (inclusive) index we wish to start from
        :param end_index: the (exclusive) index we wish to end on
        :return: orders, trades, cancels DataFrames
        """
        data_root = root
        dirs = self.get_dirs_in_dir(data_root)

        if end_index is None:
            end_index = len(dirs)

        dirs_to_load = dirs[start_index:end_index]
        return_list = []
        for directory in dirs_to_load:
            try:
                orders_path: str = data_root + directory + "/orders.csv"
                trades_path: str = data_root + directory + "/trades.csv"
                cancels_path: str = data_root + directory + "/cancels.csv"
                midprices_path: str = data_root + directory + "/midprice.csv"
                best_bids_path: str = data_root + directory + "/bestBids.csv"
                best_asks_path: str = data_root + directory + "/bestAsks.csv"

                orders_dd = self.format_dd(dd.read_csv(orders_path))
                trades_dd = self.format_dd(dd.read_csv(trades_path))
                cancels_dd = self.format_dd(dd.read_csv(cancels_path))
                midprices_dd = self.format_midprice_dd(dd.read_csv(midprices_path))
                best_bids_dd = self.format_midprice_dd(dd.read_csv(best_bids_path))
                best_asks_dd = self.format_midprice_dd(dd.read_csv(best_asks_path))

                return_list.append((orders_dd, trades_dd, cancels_dd, midprices_dd, best_bids_dd, best_asks_dd))
            except EmptyDataError:
                self.logger.info("Failed to load " + directory)
                print("Failed to load " + directory)

        return return_list

    @staticmethod
    def get_dirs_in_dir(data_root):
        dirs = next(os.walk(data_root))[1]
        return dirs

    @staticmethod
    def get_files_in_dir(data_root):
        dirs = next(os.walk(data_root))[2]
        return dirs

    @classmethod
    def load_feed(cls, root, start_time: datetime, end_time: datetime, product: str, fmt: str = "parquet") -> dd:
        """Loads in a feed of real data and applies formatting to timestamp, price and size columns"""
        # Assume data is on the same day and just hours apart for now
        hour_delta = end_time.hour - start_time.hour
        files_to_load = []

        # TODO: introduce wrapping over days
        # TODO: split this function up!
        # TODO: BUG: struggles to load small blobs of data
        for i in range(0, hour_delta + 1):
            filename = start_time.date().isoformat() + "/" + str("%02i" % (start_time.hour + i)) + "." + fmt
            cls.logger.debug(filename)
            files_to_load.append(filename)

        feed_df = pd.DataFrame()
        for filename in files_to_load:
            file_path = root + filename
            if fmt == "parquet":
                file_df = pd.read_parquet(file_path)
            else:
                file_df = pd.read_csv(file_path)
            file_df = DataSplitter.get_product(product, file_df)
            file_df = DataLoader().format_dd(file_df)
            file_df = file_df[start_time < file_df['time']]
            file_df = file_df[file_df['time'] < end_time]
            feed_df = feed_df.append(file_df)

        return feed_df

    @staticmethod
    def load_split_data(real_root, start_time, end_time, product):
        feed_df = DataLoader().load_feed(real_root, start_time, end_time, product)
        feed_df = DataSplitter.get_product(product, feed_df)

        orders_df = DataSplitter.get_orders(feed_df)
        trades_df = DataSplitter.get_trades(feed_df)
        cancels_df = DataSplitter.get_cancellations(feed_df)

        return orders_df, trades_df, cancels_df
