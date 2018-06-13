import logging
import pathlib

import dask.dataframe as dd
import pandas as pd

from data.data_splitter import DataSplitter
from data.data_transformer import DataTransformer
from data.data_utils import DataUtils
from distribution_fitter import DistributionFitter


class Graphing:
    def __init__(self, config, data_desc: str):
        self.config = config

        self.data_description = data_desc
        self.logger = logging.getLogger(self.__class__.__name__)

    def graph_interval(self, orders_df: dd):
        order_time_delta_df = orders_df['time'].apply(lambda x: DataUtils.date_to_unix(x, 'ns') / 1e6).diff()
        self.logger.debug(order_time_delta_df)
        cleaned_df = order_time_delta_df[order_time_delta_df != 0]
        self.graph_distribution(cleaned_df, self.data_description + ' inter-order arrival times',
                                'inter order time (ms)', bins=100)

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

    # TODO: calculate price % difference (so that we can compare these distributions between currencies or points in time where the price is very different
    # TODO: make this use the mid price as calculated by what the order book actually looks like
    def graph_relative_price_distribution(self, trades_df: dd, other_df: dd, num_bins=100):
        buy_orders = DataSplitter.get_side("buy", other_df)
        sell_orders = DataSplitter.get_side("sell", other_df)

        buy_prices = DataTransformer.get_relative_prices(trades_df, buy_orders)
        buy_prices = buy_prices.apply(lambda x: -x)
        sell_prices = DataTransformer.get_relative_prices(trades_df, sell_orders)

        # Graphing
        self.config.plt.figure(figsize=(12, 8))

        self.graph_distribution(buy_prices, self.data_description + ", Buy Side", "Price relative to most recent trade",
                                bins=num_bins)
        self.graph_distribution(sell_prices, self.data_description + ", Sell Side",
                                "Price relative to most recent trade", bins=num_bins)

    @staticmethod
    def __graph_price_time_set(df: dd, marker: str):
        y = df['price'].astype('float64').fillna(method='ffill')

        times = df['time'].astype('datetime64[ns]').apply(lambda x: DataUtils.date_to_unix(x, 'ns'))
        start_time = times.min()
        x = times.apply(lambda z: (z - start_time) / 1e9)

        self.config.plt.plot(x, y, marker)

    def graph_price_time(self, df: dd, data_desc: str, mid: int, ywindow: int):
        self.config.plt.figure(figsize=(12, 8))

        buy_df = DataSplitter.get_side("buy", df)
        sell_df = DataSplitter.get_side("sell", df)

        self.__graph_price_time_set(buy_df, 'r+')
        self.__graph_price_time_set(sell_df, 'b+')

        self.config.plt.xlabel('Time (s)')
        self.config.plt.ylabel('Price ($)')

        ymin, ymax = self.get_y_bounds(mid, ywindow)

        self.config.plt.ylim(ymin, ymax)
        self.config.plt.xlim(0, self.config.simulation_window)

        self.config.plt.title(self.data_description + " " + data_desc + ' price')

        return self.config.plt

    @staticmethod
    def get_y_bounds(mid, ywindow):
        ymin = mid - (ywindow / 2)
        ymax = mid + (ywindow / 2)

        return ymin, ymax

    def graph_order_cancel_relative_price_distribution(self, feed_df):
        trades_df = DataSplitter.get_trades(feed_df)
        cancels_df = DataSplitter.get_cancellations(feed_df)
        self.graph_relative_price_distribution(trades_df, cancels_df)

    # PRE: assume that the incoming df is either all trades or all orders
    # (not sure the data will make much sense otherwise)
    @staticmethod
    def graph_price_quantity(df: dd) -> None:
        prices = df['price'].astype('float64').tolist()
        quantities = df['size'].astype('float64').tolist()

        self.config.plt.figure(figsize=(12, 8))
        self.config.plt.scatter(prices, quantities, marker='+')

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

    # Source: https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
    def plot_mean_and_ci_and_real_values(self, mean, lb, ub, times, real_times, real_prices, color_mean=None,
                                         color_shading=None):
        # Set bounds and make title (+ for axes)
        self.config.plt.figure(figsize=(12, 8))
        ymin = real_prices.iloc[0] - (self.config.ywindow / 2)
        ymax = real_prices.iloc[0] + (self.config.ywindow / 2)
        self.config.plt.ylim(ymin, ymax)

        # plot the shaded range of the confidence intervals
        self.config.plt.fill_between(times, ub, lb, color=color_shading, alpha=.5,
                                     label="Simulated 2.5th and 97.5th percentiles")

        # plot the mean on top
        self.config.plt.plot(times, mean, color_mean, label="Simulated Median (50th percentile")
        self.config.plt.plot(real_times, real_prices, 'r+', label="Real Trades")
        self.config.plt.legend(loc='upper right')

    def plot_spread(self, best_bid_data, best_ask_data):
        for index in best_bid_data.keys():
            ymin, ymax = self.get_y_bounds(
                (best_bid_data[index]['price'].iloc[0] + best_ask_data[index]['price'].iloc[0]) / 2,
                self.config.ywindow)

            self.config.plt.figure(figsize=(12, 8))
            self.config.plt.ylim(ymin, ymax)

            self.config.plt.plot(best_bid_data[index]['time'], best_bid_data[index]['price'], 'g',
                                 label='Best Bid Price')
            self.config.plt.plot(best_ask_data[index]['time'], best_ask_data[index]['price'], 'r',
                                 label='Best Ask Price')
            self.config.plt.legend()
            self.config.plt.ylabel("Price ($)")
            self.config.plt.xlabel("Time (s)")
            self.config.plt.title("Spread")
            self.config.plt.show()  # TODO: make this conform to graphing mode

    # def plot_comparison(self, sim_st, sim_mid_means, sim_mid_ub, sim_mid_lb, times, real_times,
    #                     real_prices):
    #     self.config.plt.title(self.config.product + " at " + self.get_plot_title())
    #     self.config.plt.xlabel("Time (seconds)")
    #     self.config.plt.ylabel("Price ($)")
    #
    #     # plot the data
    #     self.plot_mean_and_ci_and_real_values(sim_mid_means, sim_mid_ub, sim_mid_lb, times, real_times,
    #                                           real_prices,
    #                                           color_mean='k',
    #                                           color_shading='k')
    #     self.output_graph(sim_st, "midprices", self.get_plot_title())
    #
    #     self.config.plt.close()

    def plot_comparison(self, category: str, sim_st, sim_means, sim_ub, sim_lb, times, real_times,
                        real_prices, title=None):
        if not title:
            title = self.config.product + " at " + str(sim_st)

        self.config.plt.title(title)
        self.config.plt.xlabel("Time (seconds)")
        self.config.plt.ylabel("Price ($)")

        # plot the data
        self.plot_mean_and_ci_and_real_values(sim_means, sim_ub, sim_lb, times, real_times, real_prices,
                                              color_mean='k',
                                              color_shading='k')
        self.output_graph(sim_st, category, sim_st.isoformat())
        self.config.plt.close()

    def plot_monte_carlo(self, start_price, monte_carlo_data, sim_st, times):

        self.config.plt.figure(figsize=(12, 8))
        ymin = start_price - (self.config.ywindow / 2)
        ymax = start_price + (self.config.ywindow / 2)
        self.config.plt.ylim(ymin, ymax)

        for _, prices_for_sim in monte_carlo_data.items():
            if len(prices_for_sim) == 0:
                continue
            self.config.plt.plot(times, prices_for_sim)

        self.output_graph(sim_st, "monte", sim_st.isoformat())

    def output_graph(self, sim_st, category, title: str):
        self.config.plt.title(title + " " + category)
        self.config.plt.xlabel("Time (s)")
        # TODO chnage this dollar for other quote currencies.
        self.config.plt.ylabel("Price ($)")

        if self.config.graph_mode == "save":
            plot_root = self.config.graphs_output_root + category + "/" + sim_st.date().isoformat() + "/"
            self.save_figure(plot_root, title)
        if self.config.graph_mode == "show":
            self.config.plt.show()

    def save_figure(self, plot_root, title):
        # Ensure directory exists
        pathlib.Path(plot_root).mkdir(parents=True, exist_ok=True)

        plot_path = plot_root + title + ".png"

        self.config.plt.savefig(plot_path, dpi=600, transparent=True)
        self.logger.info("Saved plot to: " + plot_path)
