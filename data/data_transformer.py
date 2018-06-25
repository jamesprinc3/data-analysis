import logging

import dask.dataframe as dd

from data.data_utils import DataUtils
from orderbook.orderbook_evolutor import OrderBookEvolutor
from stats import Statistics


class DataTransformer:

    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

    @staticmethod
    def get_relative_prices(trades_df: dd, other_df: dd) -> dd:
        price_over_time: dd = Statistics().get_price_over_time(trades_df).groupby(['time'])['most_recent_trade_price'].mean().to_frame()
        other_df = DataUtils().fuzzy_join(other_df, price_over_time, on='time')
        relative_prices = other_df['relative_price']
        relative_prices = DataUtils().remove_tails(relative_prices, 3)
        return relative_prices

    @staticmethod
    def get_prices_relative_to_midprice(ob_state_df, ob_state_seq_num, ob_state_time, feed_df, other_df) -> dd:
        order_evo = OrderBookEvolutor(ob_state_df, ob_state_time, ob_state_seq_num)
        midprices_over_time = order_evo.evolve_orderbook(feed_df)
        other_df = other_df.merge(midprices_over_time, how='left', on='time')
        other_df['relative_price'] = other_df['price'] - other_df['midprice']
        # other_df = DataUtils().fuzzy_join(other_df, midprices_over_time, on='time')
        relative_prices = other_df['relative_price'].dropna()
        relative_prices = DataUtils().remove_tails(relative_prices, 3)
        return relative_prices

    @staticmethod
    def get_time_intervals(df):
        intervals = df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns') / 1e6).diff()
        cleaned_intervals = intervals[intervals != 0].dropna()
        return cleaned_intervals
