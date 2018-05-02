from datetime import timedelta

import pandas as pd
import dask.dataframe as dd
import time
import numpy as np


class DataUtils:

    @staticmethod
    def date_to_unix(s, unit: str):
        return pd.to_datetime(s, unit=unit).value

    @staticmethod
    def keep_n_std_dev(data: pd.Series, n: int) -> pd.Series:
        return data[~((data - data.mean()).abs() > n * data.std())]

    # TODO: we can probably replace this into a few places in the codebase
    def get_times_in_seconds_after_start(self, series: pd.Series):
        # print(series)
        # series = pd.to_datetime(series, unit='ns')
        start_time = series.iloc[0]
        # print(series)
        series = series.apply(lambda x: (x - start_time))
        series = series.apply(lambda x: x.total_seconds())

        # print(series)

        return series


    @staticmethod
    def get_last_price_before(trades_df: dd, seconds: int):
        local_df = trades_df.copy()
        local_df['time'] = DataUtils().get_times_in_seconds_after_start(local_df['time'])
        # print(local_df['time'])
        trades_before = local_df[local_df['time'] < seconds]
        return trades_before['price'].iloc[-1]

