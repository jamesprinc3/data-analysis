import logging

import dask.dataframe as dd

from data.data_splitter import DataSplitter
from data.data_utils import DataUtils
from distribution_fitter import DistributionFitter
from stats import Statistics


class DataTransformer:

    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

    # Perhaps this ought to be moved somewhere
    @staticmethod
    def price_distributions(trades_df: dd, other_df: dd, bins=100, relative=True, graph=False):
        ret = {}

        for side in ["buy", "sell"]:
            side_df = DataSplitter.get_side(side, other_df)
            prices = side_df['price']

            if relative:
                prices = DataTransformer.get_relative_prices(trades_df, side_df)
                # Flip distribution for better fit
                if side == "buy":
                    prices = prices.apply(lambda x: -x)

            best_fit, best_fit_params = DistributionFitter.best_fit_distribution(prices, bins)
            best_dist, dist_str = DistributionFitter.get_distribution_string(best_fit, best_fit_params)

            data_desc = side + " prices"
            if relative:
                data_desc += " relative"

            if graph:
                DistributionFitter.plot_data_with_distribution(prices, best_dist, best_fit_params,
                                                               data_desc, "prices", show=True)

            ret[side] = (best_dist, dist_str)

        return ret

    @staticmethod
    def get_relative_prices(trades_df: dd, other_df: dd) -> dd:
        price_over_time: dd = Statistics().get_price_over_time(trades_df).groupby(['time'])['most_recent_trade_price'].mean().to_frame()
        other_df = DataUtils().fuzzy_join(other_df, price_over_time)
        relative_prices = other_df['relative_price']
        relative_prices = DataUtils().remove_tails(relative_prices, 3)
        return relative_prices


    @staticmethod
    def intervals_distribution(df: dd):
        cleaned_intervals = DataTransformer.get_time_intervals(df)

        intervals_best_fit, intervals_best_fit_params = DistributionFitter.best_fit_distribution(cleaned_intervals)
        intervals_best_dist, intervals_dist_str = DistributionFitter.get_distribution_string(intervals_best_fit, intervals_best_fit_params)

        return intervals_best_dist, intervals_dist_str

    @staticmethod
    def get_time_intervals(df):
        intervals = df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns') / 1e6).diff()
        cleaned_intervals = intervals[intervals != 0].dropna()
        return cleaned_intervals
