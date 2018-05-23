import pebble

from data.data_splitter import DataSplitter
from data.data_transformer import DataTransformer
from distribution_fitter import DistributionFitter
from stats import Statistics


class Sample:

    @staticmethod
    def generate_order_params(trades_df, orders_df, cancels_df):
        params = {}
        distributions = {}
        ratios = {}

        with pebble.ProcessPool() as pool:
            # Find distributions using different procs
            relative_order_price_distributions = pool.schedule(DataTransformer.price_distributions,
                                                               (trades_df, orders_df,),
                                                               dict(relative=True))

            # Buy/sell Price
            order_price_distributions = pool.schedule(DataTransformer.price_distributions,
                                                      (trades_df, orders_df,), dict(relative=False))

            # Buy/sell price Cancellation
            relative_cancel_price_distributions = pool.schedule(DataTransformer.price_distributions,
                                                                (trades_df, cancels_df,))

            # Limit/ Market Order Size
            limit_orders = DataSplitter.get_limit_orders(orders_df)
            limit_size = pool.schedule(DistributionFitter.best_fit_distribution,
                                       (limit_orders['size'],))

            market_orders = DataSplitter.get_market_orders(orders_df)
            market_size = pool.schedule(DistributionFitter.best_fit_distribution,
                                        (market_orders['size'],))

            intervals = pool.schedule(DataTransformer.intervals_distribution, (orders_df,))

            ratios["buy_sell_order_ratio"] = Statistics.get_buy_sell_order_ratio(orders_df)
            ratios["buy_sell_cancel_ratio"] = Statistics.get_buy_sell_order_ratio(cancels_df)
            ratios["buy_sell_volume_ratio"] = Statistics.get_buy_sell_volume_ratio(orders_df)
            ratios['limit_market_order_ratio'] = Statistics.get_limit_market_order_ratio(orders_df)

            params["distributions"] = distributions
            params['ratios'] = ratios

            # Buy/sell Price relative
            distributions["buy_price_relative"] = relative_order_price_distributions.result()["buy"][1]
            distributions["sell_price_relative"] = relative_order_price_distributions.result()["sell"][1]

            distributions["buy_price"] = order_price_distributions.result()["buy"][1]
            distributions["sell_price"] = order_price_distributions.result()["sell"][1]

            distributions["buy_cancel_price"] = relative_cancel_price_distributions.result()["buy"][1]
            distributions["sell_cancel_price"] = relative_cancel_price_distributions.result()["sell"][1]

            limit_size_best_fit, limit_size_best_fit_params = limit_size.result()
            _, distributions["limit_size"] = DistributionFitter.get_distribution_string(limit_size_best_fit,
                                                                                        limit_size_best_fit_params)

            market_size_best_fit, market_size_best_fit_params = market_size.result()
            _, distributions["market_size"] = DistributionFitter.get_distribution_string(market_size_best_fit,
                                                                                         market_size_best_fit_params)

            _, distributions["interval"] = intervals.result()

        return params
