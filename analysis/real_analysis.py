from typing import List

import matplotlib.pyplot as plt

from data_loader import DataLoader
from data_splitter import DataSplitter
from data_transformer import DataTransformer
from distribution_fitter import DistributionFitter
from graph_creator import GraphCreator
from stats import Statistics
from data_utils import DataUtils


class RealAnalysis:
    def __init__(self, file_path, data_description):
        self.file_path = file_path
        self.data_description = data_description

        self.input_dd = DataLoader().load_real_data(self.file_path)
        self.btc_usd_dd = self.input_dd[self.input_dd['product_id'] == 'BTC-USD']

    @staticmethod
    def secs_to_nanos(secs) -> int:
        return secs * 10 ** 9

    def generate_order_distributions(self):
        orders_df = Statistics.get_orders(self.btc_usd_dd).compute()
        orders_df = DataSplitter.get_last_n_nanos(orders_df, (10 * 60) * (10 ** 9))

        trades_df = Statistics.get_trades(self.btc_usd_dd).compute()
        trades_df = DataSplitter.get_last_n_nanos(trades_df, (10 * 60) * (10 ** 9))

        relative_order_price_distributions = DataTransformer.relative_price_distribution(trades_df, orders_df)

        # Buy/sell Price
        buy_price_relative_distribution = relative_order_price_distributions["buy"]
        sell_price_relative_distribution = relative_order_price_distributions["sell"]

        # Buy/sell Ratio
        buy_sell_ratio = Statistics.get_buy_sell_ratio(orders_df)

        # Buy/sell price Cancellation
        cancels_df = Statistics.get_cancellations(self.btc_usd_dd).compute()
        relative_cancel_price_distributions = DataTransformer.relative_price_distribution(trades_df, orders_df)
        buy_price_cancellation_distribution = relative_cancel_price_distributions["buy"]
        sell_price_cancellation_distribution = relative_cancel_price_distributions["sell"]

        # Quantity
        # quantity_distribution =
        quantity_best_fit, quantity_best_fit_params = DistributionFitter.best_fit_distribution(orders_df['size'])
        quantity_best_dist, quantity_dist_str = DistributionFitter.get_distribution_string(quantity_best_fit, quantity_best_fit_params)

        # Interval
        intervals_best_dist, intervals_dist_str = DataTransformer.intervals_distribution(orders_df)

        print(buy_price_relative_distribution)
        print(sell_price_relative_distribution)

        print(buy_sell_ratio)

        print(buy_price_cancellation_distribution)
        print(sell_price_cancellation_distribution)

        print(quantity_best_dist, quantity_dist_str)

        print(intervals_best_dist, intervals_dist_str)

        pass

    def generate_graphs(self):


        # btc_usd_price_buy = pd.Series(Statistics.get_side('buy', input_dd)['price'].astype('float64').tolist())
        #
        # sample_size = 100
        # std_devs = 3

        # data = Statistics.keep_n_std_dev(btc_usd_price_buy, std_devs)
        # if len(btc_usd_price_buy) > sample_size:
        #     data = btc_usd_price_buy.sample(n=sample_size)

        # TODO: include the time in the data_descriptions
        graph_creator = GraphCreator("Real BTC-USD")
        num_seconds = 10

        orders_df = Statistics.get_orders(self.btc_usd_dd).compute()
        orders_df = DataSplitter.get_last_n_nanos(orders_df, (5 * 60) * (10 ** 9))
        orders_df = DataSplitter.get_first_n_nanos(orders_df, self.secs_to_nanos(num_seconds))
        graph_creator.graph_price_time(orders_df, "orders")
        # Statistics().calculate_feed_stats(orders_df)


        # graph_creator.graph_price_time(Statistics.get_trades(btc_usd_df))
        # graph_creator.graph_order_sizes(orders_df)
        # graph_creator.graph_price_quantity(orders_df)
        # graph_creator.graph_time_delta(orders_df)

        # graph_creator.graph_order_cancel_relative_price_distribution(btc_usd_df)

        trades_df = Statistics.get_trades(self.btc_usd_dd).compute()

        trades_df = DataSplitter.get_first_n_nanos(trades_df, self.secs_to_nanos(num_seconds))
        graph_creator.graph_price_time(trades_df, "trades")

        # graph_creator.graph_sides(Statistics.get_orders(btc_usd_df))
        # Statistics().get_price_over_time(btc_usd_df)
        # graph_creator.graph_order_relative_price_distribution(btc_usd_df)
        # graph_creator.graph_time_delta(Statistics.get_orders(btc_usd_df))

        graph_creator.graph_relative_price_distribution(trades_df, orders_df, 100)

        plt.show()

    def get_prices_at_times(self, seconds_list: List[int]):
        input_dd = DataLoader().load_real_data(self.file_path)
        btc_usd_dd = input_dd[input_dd['product_id'] == 'BTC-USD']

        trades_df = btc_usd_dd.compute()

        map(lambda x: DataUtils.get_last_price_before(trades_df, x), seconds_list)

        # num_total = total(df)
        # num_btc_usd = total(btc_usd_df)
        # print("percentage of orders of this market vs whole feed: " + str((100*num_btc_usd) / num_total) + "%")
        #



        # print(df['product_id'].unique())
        # print(df)


        # print(btc_usd_df)

        # trades = stats.get_trades(btc_usd_df)[['order_id', 'price']].dropna()
        # order_sizes = stats.get_orders(btc_usd_df)[['order_id', 'size']]
        # joined = trades.join(order_sizes.set_index('order_id'), how='inner', on='order_id')
        # print(trades)
        # print(order_sizes)
        # graphs.graph_price_quantity(joined)
        # plt.show()
        # stats.calculate_stats(btc_usd_df)
        #
        # graph_sides(btc_usd_df, "BTC-USD")


        # btc_usd_price_buy = keep_n_std_dev(btc_usd_price_buy, std_devs)
        # print("here")
        #
        # theoretical, _ = fitting.best_fit_with_graphs(data, 200)
        #
        # print(theoretical)
        #
        # qqplot.plot(btc_usd_price_buy, theoretical)
        #
