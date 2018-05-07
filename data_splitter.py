import dask.dataframe as dd

from data_utils import DataUtils


class DataSplitter:
    @staticmethod
    def get_first_n_nanos(df: dd, nanos: int):
        start_time = df.iloc[0]['time']
        end_time = DataUtils.date_to_unix(start_time, 'ns') + nanos

        converted = df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns'))

        return df[converted < end_time]

    # TODO: rename to get_last(nanos=[blah])
    @staticmethod
    def get_last_n_nanos(df: dd, nanos: int):
        end_time = df.iloc[0]['time']
        start_time = DataUtils.date_to_unix(end_time, 'ns') + nanos

        converted = df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns'))

        return df[start_time > converted]

    @staticmethod
    def get_side(side: str, df: dd) -> dd:
        return df[df['side'] == side]

    @staticmethod
    def get_trades(df: dd) -> dd:
        return df[df['reason'] == 'filled']

    @staticmethod
    def get_orders(df: dd) -> dd:
        return df[df['type'] == 'received']

    @staticmethod
    def get_cancellations(df: dd) -> dd:
        return df[df['reason'] == 'canceled']

    @staticmethod
    def get_product(product: str, df: dd) -> dd:
        return df[df['product_id'] == product]
