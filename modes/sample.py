import logging
from typing import List

import numpy as np
import pandas as pd
import pebble

from correlations import Correlations
from data.data_splitter import DataSplitter
from data.data_transformer import DataTransformer
from stats import Statistics


class Sample:
    logger = logging.getLogger("Sample")

    @classmethod
    def check_has_elements(cls, dfs: List[pd.DataFrame]):
        try:
            for df in dfs:
                if df.size == 0:
                    raise ValueError("df is empty")
        except Exception as e:
            cls.logger.error(str(e))

    @staticmethod
    def plot_cdf(x, cy, data_desc: str = ""):
        import matplotlib.pyplot as plt

        plt.plot(x, cy)
        plt.title(data_desc + " cdf")

        plt.show()

    @staticmethod
    def get_cdf_data(vals):
        x = vals.sort_values()
        # Normalise y axis
        y = vals / vals.sum()
        cy = np.cumsum(y)
        return x, cy

    @classmethod
    def generate_sim_params(cls, orders_df, trades_df, cancels_df, graph=False):
        cls.check_has_elements([orders_df, trades_df, cancels_df])

        try:
            params = {}
            distributions = {}
            ratios = {}
            correlations = {}
            discrete_distributions = {}

            # TODO: reduce code duplication and parallelise inverse tingz
            with pebble.ProcessPool() as pool:
                price_size_corrs = Correlations.get_price_size_corr(trades_df, DataSplitter.get_limit_orders(orders_df))
                correlations['buy_price_size'] = price_size_corrs['buy']
                correlations['sell_price_size'] = price_size_corrs['sell']

                sell_orders = DataSplitter.get_side("sell", orders_df)
                sell_prices_relative = DataTransformer.get_relative_prices(trades_df, sell_orders)
                sell_x, sell_cy = Sample.get_cdf_data(sell_prices_relative)
                discrete_distributions["sell_price_relative"] = {'x': sell_x.tolist(), 'cy': sell_cy.tolist()}
                Sample.plot_cdf(sell_x, sell_cy, "Sell order prices (relative)")

                buy_orders = DataSplitter.get_side("buy", orders_df)
                buy_prices_relative = DataTransformer.get_relative_prices(trades_df, buy_orders)
                buy_prices_relative = buy_prices_relative.apply(lambda x: -x)
                buy_x, buy_cy = Sample.get_cdf_data(buy_prices_relative)
                discrete_distributions["buy_price_relative"] = {'x': buy_x.tolist(), 'cy': buy_cy.tolist()}
                Sample.plot_cdf(buy_x, buy_cy, "Buy prices (relative) (flipped for comparison)")

                market_orders = DataSplitter.get_market_orders(orders_df)

                buy_market_sizes = DataSplitter.get_side("buy", market_orders)['size'].dropna().apply(lambda x: abs(x))
                buy_market_sizes_x, buy_market_sizes_cy = Sample.get_cdf_data(buy_market_sizes)
                discrete_distributions["buy_market_size"] = \
                    {'x': buy_market_sizes_x.tolist(), 'cy': buy_market_sizes_cy.tolist()}
                Sample.plot_cdf(buy_x, buy_cy, "Buy market order sizes")

                sell_market_sizes = DataSplitter.get_side("sell", market_orders)['size'].dropna().apply(
                    lambda x: abs(x))
                sell_market_sizes_x, sell_market_sizes_cy = Sample.get_cdf_data(sell_market_sizes)
                discrete_distributions["sell_market_size"] = \
                    {'x': sell_market_sizes_x.tolist(), 'cy': sell_market_sizes_cy.tolist()}
                Sample.plot_cdf(sell_market_sizes_x, sell_market_sizes_cy, "Sell market order sizes")


                # Find distributions using different procs
                # relative_order_price_distributions = pool.schedule(DataTransformer.price_distributions,
                #                                                    (trades_df, orders_df,),
                #                                                    dict(relative=True, graph=graph))

                # Buy/sell Price
                # order_price_distributions = pool.schedule(DataTransformer.price_distributions,
                #                                           (trades_df, orders_df,),
                #                                           dict(relative=False, graph=True))

                # Buy/sell price Cancellation
                relative_cancel_price_distributions = pool.schedule(DataTransformer.price_distributions,
                                                                    (trades_df, cancels_df,))

                # Limit Order Size
                limit_orders = DataSplitter.get_limit_orders(orders_df)

                buy_limit_orders_size = DataSplitter.get_side("buy", limit_orders)['size'].dropna().apply(
                    lambda x: abs(x))
                buy_limit_order_sizes_x, buy_limit_order_sizes_cy = Sample.get_cdf_data(buy_limit_orders_size)
                discrete_distributions["buy_limit_size"] = \
                    {'x': buy_limit_order_sizes_x.tolist(), 'cy': buy_limit_order_sizes_cy.tolist()}
                Sample.plot_cdf(buy_limit_order_sizes_x, buy_limit_order_sizes_cy, "Buy limit order sizes")

                sell_limit_orders_size = DataSplitter.get_side("sell", limit_orders)['size'].dropna().apply(
                    lambda x: abs(x))
                sell_limit_order_sizes_x, sell_limit_order_sizes_cy = Sample.get_cdf_data(sell_limit_orders_size)
                discrete_distributions["sell_limit_size"] = \
                    {'x': sell_limit_order_sizes_x.tolist(), 'cy': sell_limit_order_sizes_cy.tolist()}
                Sample.plot_cdf(sell_limit_order_sizes_x, sell_limit_order_sizes_cy, "Sell limit order sizes")

                intervals = DataTransformer.get_time_intervals(orders_df)
                intervals_x, intervals_cy = Sample.get_cdf_data(intervals)
                discrete_distributions["intervals"] = \
                    {'x': intervals_x.tolist(), 'cy': intervals_cy.tolist()}
                Sample.plot_cdf(intervals_x, intervals_cy, "Order intervals")

                # buy_limit_size = pool.schedule(DistributionFitter.best_fit_distribution,
                #                                (buy_limit_orders['size'],))
                # sell_limit_size = pool.schedule(DistributionFitter.best_fit_distribution,
                #                                 (sell_limit_orders['size'],))

                # Market Order Size

                # market_orders = DataSplitter.get_market_orders(orders_df)
                # buy_market_orders = DataSplitter.get_side("buy", market_orders)
                # sell_market_orders = DataSplitter.get_side("sell", market_orders)

                # buy_market_size = pool.schedule(DistributionFitter.best_fit_distribution,
                #                                (buy_market_orders['size'],))
                # sell_market_size = pool.schedule(DistributionFitter.best_fit_distribution,
                #                                 (sell_market_orders['size'],))

                # intervals = pool.schedule(DataTransformer.intervals_distribution, (orders_df,))

                ratios["buy_sell_order_ratio"] = Statistics.get_buy_sell_ratio(orders_df)
                ratios["buy_sell_cancel_ratio"] = Statistics.get_buy_sell_ratio(cancels_df)
                ratios["buy_sell_volume_ratio"] = Statistics.get_buy_sell_volume_ratio(orders_df)
                ratios['limit_market_order_ratio'] = Statistics.get_limit_market_order_ratio(orders_df)

                # Buy/sell Price relative
                # distributions["buy_price_relative"] = relative_order_price_distributions.result()["buy"][1]
                # distributions["sell_price_relative"] = relative_order_price_distributions.result()["sell"][1]

                # distributions["buy_price"] = order_price_distributions.result()["buy"][1]
                # distributions["sell_price"] = order_price_distributions.result()["sell"][1]

                # distributions["buy_cancel_price"] = relative_cancel_price_distributions.result()["buy"][1]
                # distributions["sell_cancel_price"] = relative_cancel_price_distributions.result()["sell"][1]

                # buy_limit_size_best_fit, buy_limit_size_best_fit_params = buy_limit_size.result()
                # _, distributions["buy_limit_size"] = DistributionFitter.get_distribution_string(buy_limit_size_best_fit,
                #                                                                                 buy_limit_size_best_fit_params)
                #
                # sell_limit_size_best_fit, sell_limit_size_best_fit_params = sell_limit_size.result()
                # _, distributions["sell_limit_size"] = DistributionFitter.get_distribution_string(sell_limit_size_best_fit,
                #                                                                                  sell_limit_size_best_fit_params)

                # buy_market_size_best_fit, buy_market_size_best_fit_params = buy_market_size.result()
                # _, distributions["buy_market_size"] = DistributionFitter.get_distribution_string(buy_market_size_best_fit,
                #                                                                                  buy_market_size_best_fit_params)
                #
                # sell_market_size_best_fit, sell_market_size_best_fit_params = sell_market_size.result()
                # _, distributions["sell_market_size"] = DistributionFitter.get_distribution_string(sell_market_size_best_fit,
                #                                                                                   sell_market_size_best_fit_params)

                # _, distributions["interval"] = intervals.result()

                params['ratios'] = ratios
                params['correlations'] = correlations
                params['distributions'] = distributions
                params['discreteDistributions'] = discrete_distributions

            return params
        except Exception as e:
            cls.logger.error("Failed to generate parameters, exception was " + str(e))
            raise e
