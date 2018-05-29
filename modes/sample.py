import pebble

from correlations import Correlations
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
        correlations = {}

        with pebble.ProcessPool() as pool:
            price_size_corrs = Correlations.get_price_size_corr(trades_df, DataSplitter.get_limit_orders(orders_df))
            correlations['buy_price_size'] = price_size_corrs['buy']
            correlations['sell_price_size'] = price_size_corrs['sell']

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

            # Limit Order Size
            limit_orders = DataSplitter.get_limit_orders(orders_df)
            buy_limit_orders = DataSplitter.get_side("buy", limit_orders)
            sell_limit_orders = DataSplitter.get_side("sell", limit_orders)

            buy_limit_size = pool.schedule(DistributionFitter.best_fit_distribution,
                                           (buy_limit_orders['size'],))
            sell_limit_size = pool.schedule(DistributionFitter.best_fit_distribution,
                                            (sell_limit_orders['size'],))

            # Market Order Size

            market_orders = DataSplitter.get_market_orders(orders_df)
            buy_market_orders = DataSplitter.get_side("buy", market_orders)
            sell_market_orders = DataSplitter.get_side("sell", market_orders)

            buy_market_size = pool.schedule(DistributionFitter.best_fit_distribution,
                                           (buy_market_orders['size'],))
            sell_market_size = pool.schedule(DistributionFitter.best_fit_distribution,
                                            (sell_market_orders['size'],))

            intervals = pool.schedule(DataTransformer.intervals_distribution, (orders_df,))

            ratios["buy_sell_order_ratio"] = Statistics.get_buy_sell_order_ratio(orders_df)
            ratios["buy_sell_cancel_ratio"] = Statistics.get_buy_sell_order_ratio(cancels_df)
            ratios["buy_sell_volume_ratio"] = Statistics.get_buy_sell_volume_ratio(orders_df)
            ratios['limit_market_order_ratio'] = Statistics.get_limit_market_order_ratio(orders_df)

            # Buy/sell Price relative
            distributions["buy_price_relative"] = relative_order_price_distributions.result()["buy"][1]
            distributions["sell_price_relative"] = relative_order_price_distributions.result()["sell"][1]

            distributions["buy_price"] = order_price_distributions.result()["buy"][1]
            distributions["sell_price"] = order_price_distributions.result()["sell"][1]

            distributions["buy_cancel_price"] = relative_cancel_price_distributions.result()["buy"][1]
            distributions["sell_cancel_price"] = relative_cancel_price_distributions.result()["sell"][1]

            buy_limit_size_best_fit, buy_limit_size_best_fit_params = buy_limit_size.result()
            _, distributions["buy_limit_size"] = DistributionFitter.get_distribution_string(buy_limit_size_best_fit,
                                                                                            buy_limit_size_best_fit_params)

            sell_limit_size_best_fit, sell_limit_size_best_fit_params = sell_limit_size.result()
            _, distributions["sell_limit_size"] = DistributionFitter.get_distribution_string(sell_limit_size_best_fit,
                                                                                             sell_limit_size_best_fit_params)

            buy_market_size_best_fit, buy_market_size_best_fit_params = buy_market_size.result()
            _, distributions["buy_market_size"] = DistributionFitter.get_distribution_string(buy_market_size_best_fit,
                                                                                             buy_market_size_best_fit_params)

            sell_market_size_best_fit, sell_market_size_best_fit_params = sell_market_size.result()
            _, distributions["sell_market_size"] = DistributionFitter.get_distribution_string(sell_market_size_best_fit,
                                                                                              sell_market_size_best_fit_params)

            _, distributions["interval"] = intervals.result()

            params['ratios'] = ratios
            params['correlations'] = correlations
            params['distributions'] = distributions

        return params
