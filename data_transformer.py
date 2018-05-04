from data_splitter import DataSplitter
from data_utils import DataUtils
import dask.dataframe as dd

from distribution_fitter import DistributionFitter
from stats import Statistics


class DataTransformer:

    def __init__(self):
        pass

    @staticmethod
    def relative_price_distribution(trades_df: dd, other_df: dd, bins=100):
        price_over_time: dd = Statistics().get_price_over_time(trades_df).groupby(['time'])['most_recent_trade_price'].mean().to_frame()

        # Buy Side
        buy_df = DataSplitter.get_side("buy", other_df)
        buy_df = DataUtils().fuzzy_join(buy_df, price_over_time)
        # Flip the distribution around so that we can actually fit it to something breeze can sample from
        realtive_buy_prices = buy_df['relative_price'].apply(lambda x: -x)
        logger.debug(realtive_buy_prices)
        realtive_buy_prices = DataUtils().remove_tails(realtive_buy_prices, 3)
        buy_best_fit, buy_best_fit_params = DistributionFitter.best_fit_distribution(realtive_buy_prices, bins)
        buy_best_dist, buy_dist_str = DistributionFitter.get_distribution_string(buy_best_fit, buy_best_fit_params)

        # Sell Side
        sell_df = DataSplitter.get_side("sell", other_df)
        sell_df = DataUtils().fuzzy_join(sell_df, price_over_time)
        relative_sell_prices = sell_df['relative_price']
        relative_sell_prices = DataUtils().remove_tails(relative_sell_prices, 3)
        sell_best_fit, sell_best_fit_params = DistributionFitter.best_fit_distribution(relative_sell_prices, bins)
        sell_best_dist, sell_dist_str = DistributionFitter.get_distribution_string(sell_best_fit, sell_best_fit_params)

        return {"buy": (buy_best_dist, buy_dist_str), "sell": (sell_best_dist, sell_dist_str)}

    @staticmethod
    def intervals_distribution(df: dd):
        intervals = df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns') / 1e6).diff()
        logger.debug(intervals)
        cleaned_intervals = intervals[intervals != 0]

        intervals_best_fit, intervals_best_fit_params = DistributionFitter.best_fit_distribution(cleaned_intervals)
        intervals_best_dist, intervals_dist_str = DistributionFitter.get_distribution_string(intervals_best_fit, intervals_best_fit_params)

        return intervals_best_dist, intervals_dist_str
