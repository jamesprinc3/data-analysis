import datetime
import json

from analysis.simulation_analysis import SimulationAnalysis
from analysis.real_analysis import RealAnalysis
from data_loader import DataLoader
from data_splitter import DataSplitter
from data_utils import DataUtils
from stats import Statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta


class CombinedAnalysis:
    def __init__(self, sim_root: str, real_root: str, start_time: datetime.datetime, sampling_window: int, simulation_window: int):
        """
        Class which produces validation between the simulated data and the real data
        :param sim_root: root path of the simulated data
        :param real_root: root path of the real data
        :param start_time: datetime which
        :param sampling_window: number of seconds before start_time to sample from
        :param simulation_window: number of seconds after start_time to simulate
        """
        self.sim_analysis = SimulationAnalysis(sim_root, "Combined Analysis")

        self.real_root = real_root
        self.start_time = start_time
        print(start_time)
        self.sampling_window = sampling_window
        self.simulation_window = simulation_window

    def run_simulation(self):
        orders_df, trades_df, cancels_df = self.__fetch_sampling_data()
        real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined BTC-USD")

        distributions = real_analysis.generate_order_distributions()

        self.__distributions_to_file(distributions, "/Users/jamesprince/project-data/distributions.json")

        pass


    def graph_real_prices_with_simulated_confidence_intervals(self):
        interval = 10  # seconds
        times = list(range(interval, self.simulation_window, interval))
        confidence_intervals = self.sim_analysis.calculate_confidence_at_times(times)
        print(confidence_intervals)

        real_times, real_prices = self.__fetch_real_data()

        start_price = real_prices.iloc[0]

        # Add the initial price and then convert to numpy arrays
        mean0 = np.array([start_price] + list(map(lambda lu: (lu[0] + lu[1]) / 2, confidence_intervals.values())))
        ub0 = np.array([start_price] + list(map(lambda x: x[1], confidence_intervals.values())))
        lb0 = np.array([start_price] + list(map(lambda x: x[0], confidence_intervals.values())))

        times = [0] + times

        # plot the data
        self.__plot_mean_and_CI_and_real_values(mean0, ub0, lb0, times, real_times, real_prices, color_mean='k',
                                         color_shading='k')

        plt.show()

    # Source: https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
    def __plot_mean_and_CI_and_real_values(self, mean, lb, ub, times, real_times, real_prices, color_mean=None,
                                         color_shading=None):
        # plot the shaded range of the confidence intervals
        plt.figure(figsize=(12, 8))
        plt.ylim(8000, 9000)
        plt.title("BTC-USD Price prediction")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Price ($)")
        plt.fill_between(times, ub, lb,
                         color=color_shading, alpha=.5)
        # plot the mean on top
        plt.plot(times, mean, color_mean, label="Simulated")
        plt.plot(real_times, real_prices, 'r+', label="Real")
        plt.legend()

    def __fetch_real_data(self):
        df = DataLoader().load_real_data(self.real_root, self.start_time, self.start_time + timedelta(seconds=self.simulation_window))[['time', 'price', 'reason']].compute()
        trades_df = DataSplitter.get_trades(df)

        trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
        real_times = trades_df['time']
        real_prices = trades_df['price']

        return real_times, real_prices

    def __fetch_sampling_data(self):
        feed_df = DataLoader().load_real_data(self.real_root, self.start_time - timedelta(seconds=self.simulation_window), self.start_time)

        orders_df = DataSplitter.get_orders(feed_df)
        trades_df = DataSplitter.get_trades(feed_df)
        cancels_df = DataSplitter.get_cancellations(feed_df)

        return orders_df, trades_df, cancels_df

    @staticmethod
    def __distributions_to_file(distributions: dict, file_path: str):
        with open(file_path, 'w') as fp:
            json.dump(distributions, fp)

