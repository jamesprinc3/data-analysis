import logging

from data_splitter import DataSplitter
from data_utils import DataUtils
import dask.dataframe as dd

from distribution_fitter import DistributionFitter
from stats import Statistics


class DataTransformer:

    def __init__(self):
        self.logger = logging.getLogger()

    # Perhaps this ought to be moved somewhere
    @staticmethod
    def relative_price_distributions(trades_df: dd, other_df: dd, bins=100):
        # Buy Side
        buy_df = DataSplitter.get_side("buy", other_df)
        relative_buy_prices = DataTransformer.get_relative_prices(trades_df, buy_df)
        # Flip distribution for better fit
        relative_buy_prices = relative_buy_prices.apply(lambda x: -x)
        print(relative_buy_prices)
        buy_best_fit, buy_best_fit_params = DistributionFitter.best_fit_distribution(relative_buy_prices, bins)
        buy_best_dist, buy_dist_str = DistributionFitter.get_distribution_string(buy_best_fit, buy_best_fit_params)

        # Sell Side
        sell_df = DataSplitter.get_side("sell", other_df)
        relative_sell_prices = DataTransformer.get_relative_prices(trades_df, sell_df)
        sell_best_fit, sell_best_fit_params = DistributionFitter.best_fit_distribution(relative_sell_prices, bins)
        sell_best_dist, sell_dist_str = DistributionFitter.get_distribution_string(sell_best_fit, sell_best_fit_params)

        return {"buy": (buy_best_dist, buy_dist_str), "sell": (sell_best_dist, sell_dist_str)}

    @staticmethod
    def get_relative_prices(trades_df: dd, other_df: dd) -> dd:
        price_over_time: dd = Statistics().get_price_over_time(trades_df).groupby(['time'])['most_recent_trade_price'].mean().to_frame()
        other_df = DataUtils().fuzzy_join(other_df, price_over_time)
        relative_prices = other_df['relative_price']
        relative_prices = DataUtils().remove_tails(relative_prices, 3)
        return relative_prices


    @staticmethod
    def intervals_distribution(df: dd):
        intervals = df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns') / 1e6).diff()
        cleaned_intervals = intervals[intervals != 0]

        intervals_best_fit, intervals_best_fit_params = DistributionFitter.best_fit_distribution(cleaned_intervals)
        intervals_best_dist, intervals_dist_str = DistributionFitter.get_distribution_string(intervals_best_fit, intervals_best_fit_params)

        return intervals_best_dist, intervals_dist_str
