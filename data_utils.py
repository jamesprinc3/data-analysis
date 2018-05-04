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
        # logger.debug(series)
        # series = pd.to_datetime(series, unit='ns')
        start_time = series.iloc[0]
        # logger.debug(series)
        series = series.apply(lambda x: (x - start_time))
        series = series.apply(lambda x: x.total_seconds())

        # logger.debug(series)

        return series


    @staticmethod
    def get_last_price_before(trades_df: dd, seconds: int):
        local_df = trades_df.copy()
        local_df['time'] = DataUtils().get_times_in_seconds_after_start(local_df['time'])
        # logger.debug(local_df['time'])
        trades_before = local_df[local_df['time'] < seconds]
        return trades_before['price'].iloc[-1]

    def remove_tails(self, data: dd, std_devs: int, sample_size: int=10000):
        data = DataUtils().keep_n_std_dev(data, std_devs)
        if len(data) > sample_size:
            data = data.sample(n=sample_size)
        data = DataUtils().keep_n_std_dev(data, std_devs)

        return data

    @staticmethod
    def fuzzy_join(orders: dd, price_over_time: dd) -> dd:
        orders['price'] = orders['price'].astype('float64')
        orders['time'] = orders['time'].astype('datetime64[ns]')

        price_over_time = price_over_time.reindex(orders['time'].unique(), method='nearest')

        joined = orders.join(price_over_time, on='time').fillna(method='ffill')

        joined['relative_price'] = joined.apply(
            lambda row: float(row['price']) - float(row['most_recent_trade_price']),
            axis=1)

        return joined

