import dask.dataframe as dd

from data_utils import DataUtils


class DataSplitter:
    @staticmethod
    def get_first_n_nanos(df: dd, nanos: int):
        start_time = df.iloc[0]['time']
        end_time = DataUtils.date_to_unix(start_time, 'ns') + nanos

        converted = df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns'))

        return df[converted < end_time]
