import json
from typing import List

import dask.dataframe as dd
import matplotlib.pyplot as plt

from data_splitter import DataSplitter
from data_transformer import DataTransformer
from data_utils import DataUtils
from distribution_fitter import DistributionFitter
from graph_creator import GraphCreator
from stats import Statistics


class RealAnalysis:
    def __init__(self, orders_df: dd, trades_df: dd, cancels_df: dd, data_description: str):
        if orders_df.empty or trades_df.empty or cancels_df.empty:
            raise AssertionError("At least one DataFrame is empty")

        self.orders_df = orders_df
        self.trades_df = trades_df
        self.cancels_df = cancels_df

        self.data_description = data_description

    @staticmethod
    def secs_to_nanos(secs) -> int:
        return secs * 10 ** 9

    @staticmethod
    def params_to_file(params: dict, file_path: str):
        with open(file_path, 'w') as fp:
            json.dump(params, fp)

    def generate_order_params(self):
        params = {}
        distributions = {}
        relative_order_price_distributions = DataTransformer.price_distributions(self.trades_df, self.orders_df,
                                                                                 relative=True)

        # Buy/sell Price relative
        distributions["buy_price_relative"] = relative_order_price_distributions["buy"][1]
        distributions["sell_price_relative"] = relative_order_price_distributions["sell"][1]

        # Buy/sell Price
        order_price_distributions = DataTransformer.price_distributions(self.trades_df, self.orders_df, relative=False)
        distributions["buy_price"] = order_price_distributions["buy"][1]
        distributions["sell_price"] = order_price_distributions["sell"][1]

        # Buy/sell price Cancellation
        relative_cancel_price_distributions = DataTransformer.price_distributions(self.trades_df,
                                                                                  self.cancels_df)
        distributions["buy_cancel_price"] = relative_cancel_price_distributions["buy"][1]
        distributions["sell_cancel_price"] = relative_cancel_price_distributions["sell"][1]

        # Limit/ Market Order Size
        limit_orders = DataSplitter.get_limit_orders(self.orders_df)
        limit_size_best_fit, limit_size_best_fit_params = DistributionFitter.best_fit_distribution(limit_orders['size'])
        _, distributions["limit_size"] = DistributionFitter.get_distribution_string(limit_size_best_fit,
                                                                                    limit_size_best_fit_params)

        market_orders = DataSplitter.get_market_orders(self.orders_df)
        market_size_best_fit, market_size_best_fit_params = DistributionFitter.best_fit_distribution(
            market_orders['size'])
        _, distributions["market_size"] = DistributionFitter.get_distribution_string(market_size_best_fit,
                                                                                     market_size_best_fit_params)

        # Interval
        _, distributions["interval"] = DataTransformer.intervals_distribution(self.orders_df)

        params["distributions"] = distributions

        # Buy/sell Order Ratio
        params["buy_sell_order_ratio"] = Statistics.get_buy_sell_order_ratio(self.orders_df)

        # Buy/sell Order Ratio
        params["buy_sell_cancel_ratio"] = Statistics.get_buy_sell_order_ratio(self.cancels_df)

        # Buy/sell Volume Ratio
        params["buy_sell_volume_ratio"] = Statistics.get_buy_sell_volume_ratio(self.orders_df)

        params['limit_market_order_ratio'] = Statistics.get_limit_market_order_ratio(self.orders_df)

        return params

    def generate_graphs(self, graph_root: str = None):
        # btc_usd_price_buy = pd.Series(Statistics.get_side('buy', input_dd)['price'].astype('float64').tolist())
        #
        # sample_size = 100
        # std_devs = 3

        # data = Statistics.keep_n_std_dev(btc_usd_price_buy, std_devs)
        # if len(btc_usd_price_buy) > sample_size:
        #     data = btc_usd_price_buy.sample(n=sample_size)

        # TODO: include the time in the data_descriptions
        graph_creator = GraphCreator("Real BTC-USD")

        # graph_creator.graph_sides(self.orders_df)
        # graph_creator.graph_relative_price_distribution(self.trades_df, self.orders_df, 100)
        # graph_creator.graph_interval(self.orders_df)
        graph_creator.graph_price_time(self.trades_df, "Price over time")

        plt.show()

    def get_prices_at_times(self, seconds_list: List[int]):
        map(lambda x: DataUtils.get_last_price_before(self.trades_df, x), seconds_list)

        # num_total = total(df)
        # num_btc_usd = total(btc_usd_df)
        # logger.debug("percentage of orders of this market vs whole feed: " + str((100*num_btc_usd) / num_total) + "%")
        #



        # logger.debug(df['product_id'].unique())
        # logger.debug(df)


        # logger.debug(btc_usd_df)

        # trades = stats.get_trades(btc_usd_df)[['order_id', 'price']].dropna()
        # order_sizes = stats.get_orders(btc_usd_df)[['order_id', 'size']]
        # joined = trades.join(order_sizes.set_index('order_id'), how='inner', on='order_id')
        # logger.debug(trades)
        # logger.debug(order_sizes)
        # graphs.graph_price_quantity(joined)
        # plt.show()
        # stats.calculate_stats(btc_usd_df)
        #
        # graph_sides(btc_usd_df, "BTC-USD")


        # btc_usd_price_buy = keep_n_std_dev(btc_usd_price_buy, std_devs)
        # logger.debug("here")
        #
        # theoretical, _ = fitting.best_fit_with_graphs(data, 200)
        #
        # logger.debug(theoretical)
        #
        # qqplot.plot(btc_usd_price_buy, theoretical)
        #
