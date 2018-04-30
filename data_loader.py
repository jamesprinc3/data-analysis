import dask.dataframe as dd
import os


class DataLoader:
    """Loads data and formats it appropriately so that we can always assume
    timestamps and floats are not string types to avoid silly conversions all over the place"""

    def format_dd(self, df: dd) -> dd:
        df['time'] = df['time'].astype('datetime64[ns]')
        df['price'] = df['price'].astype('float64')
        df['size'] = df['size'].astype('float64')

        return df

    def load_sim_data(self, root) -> (dd, dd, dd):
        data_root = root
        dirs = next(os.walk(data_root))[1]

        # TODO: figure out what we want to do for multiple simulations!
        # for directory in dirs:
        directory = dirs[0]
        orders_path: str = data_root + directory + "/orders.csv"
        trades_path: str = data_root + directory + "/trades.csv"
        cancels_path: str = data_root + directory + "/cancels.csv"

        print(orders_path)

        orders_dd = self.format_dd(dd.read_csv(orders_path))
        trades_dd = self.format_dd(dd.read_csv(trades_path))
        cancels_dd = self.format_dd(dd.read_csv(cancels_path))

        return orders_dd, trades_dd, cancels_dd

    def load_real_data(self, path) -> dd:
        """Loads in a feed of real data and applies formatting to timestamp, price and size columns"""
        feed_dd = dd.read_parquet(path)
        feed_dd = self.format_dd(feed_dd)

        return feed_dd
