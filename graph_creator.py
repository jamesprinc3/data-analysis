import logging

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd

from data_splitter import DataSplitter
from data_transformer import DataTransformer
from data_utils import DataUtils
from distribution_fitter import DistributionFitter


class GraphCreator:
    def __init__(self, data_desc: str):
        self.data_description = data_desc
        self.logger = logging.getLogger()

    def graph_interval(self, orders_df: dd):
        order_time_delta_df = orders_df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns') / 1e6).diff()
        self.logger.debug(order_time_delta_df)
        cleaned_df = order_time_delta_df[order_time_delta_df != 0]
        self.graph_distribution(cleaned_df, self.data_description + ' inter-order arrival times', 'inter order time (ms)', bins=100)

    def graph_order_sizes(self, orders_df: dd):
        size_df = pd.Series(orders_df['size'].astype('float64').tolist())
        self.graph_distribution(size_df, self.data_description + ' Order size', 'Order Size', 10)

    def graph_trade_sizes(self, trades_df: dd):
        order_id_map = trades_df[['order_id', 'size']].drop_duplicates()
        traded_order_id_set = pd.DataFrame(trades_df['order_id'].unique(), columns=['order_id'])
        joined = order_id_map.join(traded_order_id_set.set_index('order_id'), on='order_id', how='inner')
        result = pd.Series(joined.dropna(axis=0, how='any').reset_index(drop=True)['size'].astype('float64').tolist())
        self.graph_distribution(result, self.data_description + ' Trade Order Size', 'Trade Order Size')

    def graph_sides(self, df: dd) -> None:
        btc_usd_price_buy = pd.Series(DataSplitter.get_side('buy', df)['price'].astype('float64').tolist())
        btc_usd_price_sell = pd.Series(DataSplitter.get_side('sell', df)['price'].astype('float64').tolist())

        self.graph_distribution(btc_usd_price_buy, self.data_description + ' buy side', 'Price ($)', bins=50)
        self.graph_distribution(btc_usd_price_sell, self.data_description + ' sell side', 'Price ($)', bins=50)

    # TODO: REFACTOR (mostly replaced by data_utils.fuzzy() )
    # def format_orders(self, orders: dd, price_over_time: dd):
    #     orders['price'] = orders['price'].astype('float64')
    #     orders['time'] = orders['time'].astype('datetime64[ns]')
    #
    #     price_over_time = price_over_time.reindex(orders['time'].unique(), method='nearest')
    #
    #     # logger.debug("orders\n")
    #     # logger.debug(orders)
    #     # logger.debug("price over time\n")
    #     # logger.debug(price_over_time)
    #
    #     # B1 = orders.set_index('time').reindex(price_over_time.index, method='nearest').reset_index()
    #     joined = orders.join(price_over_time, on='time').fillna(method='ffill')
    #
    #     joined['relative_price'] = joined.apply(lambda row: float(row['price']) - float(row['most_recent_trade_price']),
    #                                             axis=1)
    #
    #     return joined

    # TODO: calculate price % difference (so that we can compare these distributions between currencies or points in time where the price is very different
    # TODO: make this use the mid price as calculated by what the order book actually looks like
    def graph_relative_price_distribution(self, trades_df: dd, other_df: dd, num_bins=100):

        buy_orders = DataSplitter.get_side("buy", other_df)
        sell_orders = DataSplitter.get_side("sell", other_df)

        buy_prices = DataTransformer.get_relative_prices(trades_df, buy_orders)
        buy_prices = buy_prices.apply(lambda x: -x)
        sell_prices = DataTransformer.get_relative_prices(trades_df, sell_orders)

        # Graphing
        plt.figure(figsize=(12, 8))

        self.graph_distribution(buy_prices, self.data_description + ", Buy Side", "Price relative to most recent trade", bins=num_bins)
        self.graph_distribution(sell_prices, self.data_description + ", Sell Side", "Price relative to most recent trade", bins=num_bins)

    def graph_price_time(self, df: dd, data_desc: str):
        #
        y = df['price'].astype('float64').fillna(method='ffill')
        self.logger.debug(y)
        self.logger.debug(df['time'].astype('datetime64[ns]'))
        times = df['time'].astype('datetime64[ns]').apply(lambda x: DataUtils.date_to_unix(x, 'ns'))
        start_time = times.min()
        self.logger.debug("start_time " + str(start_time))
        x = times.apply(lambda z: (z - start_time) / 1e6)
        # logger.debug(x)

        plt.figure(figsize=(12, 8))

        plt.plot(x, y, 'r+')

        plt.xlabel('Time (ms)')
        plt.ylabel('Price ($)')
        # plt.ylim(7000, 10000)

        plt.title(self.data_description + " " + data_desc + ' price')

    def graph_order_cancel_relative_price_distribution(self, feed_df):
        trades_df = DataSplitter.get_trades(feed_df)
        cancels_df = DataSplitter.get_cancellations(feed_df)
        self.graph_relative_price_distribution(trades_df, cancels_df)


        # price_over_time: dd = Statistics().get_price_over_time(feed_df).groupby(['time'])['most_recent_trade_price'].mean().to_frame()
        # price_over_time.index = price_over_time.index.astype('datetime64')
        #
        # cancels = Statistics.get_cancellations(feed_df)
        #
        # buy_cancels = Statistics().get_side("buy", cancels)
        # buy_cancels = self.format_orders(buy_cancels, price_over_time)
        # # Flip the distribution around so that we can actually fit it to something breeze can sample from
        # buy_cancels['relative_price'] = buy_cancels['relative_price'].apply(lambda x: -x)
        #
        # sell_cancels = Statistics().get_side("sell", cancels)
        # sell_cancels = self.format_orders(sell_cancels, price_over_time)
        #
        # self.get_distribution(buy_cancels['relative_price'], "BTC-USD Buy Side", "Relative price for order cancellations")
        # self.get_distribution(sell_cancels['relative_price'], "BTC-USD Sell Side",
        #                       "Relative price for order cancellations")

        # cancels = self.format_orders(cancels, price_over_time)
        #
        # self.get_distribution(cancels['relative_price'], 'BTC-USD', "Relative price (combined)")

    # PRE: assume that the incoming df is either all trades or all orders (not sure the data will make much sense
    # otherwise
    @staticmethod
    def graph_price_quantity(df: dd) -> None:
        prices = df['price'].astype('float64').tolist()
        quantities = df['size'].astype('float64').tolist()

        plt.figure(figsize=(12, 8))
        plt.scatter(prices, quantities, marker='+')

    @staticmethod
    def date_to_unix(s, unit: str):
        return pd.to_datetime(s, unit=unit).value

    # TODO: REFACTOR (mostly replaced by data_utils.remove_tails() )
    @staticmethod
    def graph_distribution(data: pd.Series, description: str, xlabel: str, bins=20, std_devs: int = 2):
        sample_size = 10000

        data = DataUtils().keep_n_std_dev(data, std_devs)
        if len(data) > sample_size:
            data = data.sample(n=sample_size)
        data = DataUtils().keep_n_std_dev(data, std_devs)

        DistributionFitter().best_fit_with_graphs(data, description, xlabel, bins=bins)
