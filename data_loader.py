import dask.dataframe as dd
import os
import pandas as pd

from datetime import datetime, timedelta


class DataLoader:
    """Loads data and formats it appropriately so that we can always assume
    timestamps and floats are not string types to avoid silly conversions all over the place"""

    def format_dd(self, df: dd) -> dd:
        df['time'] = df['time'].astype('datetime64[ns]')
        df['price'] = df['price'].astype('float64')
        df['size'] = df['size'].astype('float64')

        return df

    def load_sim_data(self, root, start_index=0, end_index=1) -> (dd, dd, dd):
        """

        :param root: directory that contains the output from OrderBookSimulator
        :param start_index: the (inclusive) index we wish to start from
        :param end_index: the (exclusive) index we wish to end on
        :return:
        """
        data_root = root
        dirs = next(os.walk(data_root))[1]

        # TODO: figure out what we want to do for multiple simulations!
        # for directory in dirs:
        dirs_to_load = dirs[start_index:end_index]
        return_list = []
        for directory in dirs_to_load:
            orders_path: str = data_root + directory + "/orders.csv"
            trades_path: str = data_root + directory + "/trades.csv"
            cancels_path: str = data_root + directory + "/cancels.csv"

            print(orders_path)

            orders_dd = self.format_dd(dd.read_csv(orders_path))
            trades_dd = self.format_dd(dd.read_csv(trades_path))
            cancels_dd = self.format_dd(dd.read_csv(cancels_path))

            return_list.append((orders_dd, trades_dd, cancels_dd))

        return return_list

    def load_real_data(self, root, start_time: datetime, end_time: datetime) -> dd:
        """Loads in a feed of real data and applies formatting to timestamp, price and size columns"""
        # Assume data is on the same day and just hours apart for now
        hour_delta = end_time.hour - start_time.hour
        files_to_load = []

        for i in range(0, hour_delta + 1):
            filename = start_time.date().isoformat() + "/" + str(start_time.hour + i) + ".parquet"
            print(filename)
            files_to_load.append(filename)

        feed_df = pd.DataFrame()
        for filename in files_to_load:
            file_path = root + filename
            file_df = pd.read_parquet(file_path)
            file_df = DataLoader().format_dd(file_df)
            file_df = file_df[start_time < file_df['time']]
            file_df = file_df[file_df['time'] < end_time]
            print(file_df)
            feed_df = feed_df.append(file_df)

        return feed_df
