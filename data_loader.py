import dask.dataframe as dd

class DataLoader:
    """Loads data and formats it appropriately so that we can always assume
    timestamps and floats are not string types to avoid silly conversions all over the place"""

    def load_sim_data(self, path):
        pass

    @staticmethod
    def load_real_data(path) -> dd:
        """Loads in a feed of real data and applies formatting to timestamp, price and size columns"""
        feed_dd = dd.read_parquet(path)
        feed_dd['time'] = feed_dd['time'].astype('datetime64[ns]')
        feed_dd['price'] = feed_dd['price'].astype('float64')
        feed_dd['size'] = feed_dd['size'].astype('float64')

        return feed_dd
