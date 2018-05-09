import datetime
import logging
import subprocess
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np

from analysis.real_analysis import RealAnalysis
from analysis.simulation_analysis import SimulationAnalysis
from data_loader import DataLoader
from data_splitter import DataSplitter
from data_utils import DataUtils
from graph_creator import GraphCreator
from orderbook import OrderBook
from sim_config import SimConfig
from stats import Statistics


class CombinedAnalysis:
    def __init__(self, sim_root: str, real_root: str, start_time: datetime.datetime, sampling_window: int,
                 simulation_window: int, orderbook_window: int, product: str, params_path: str):
        """
        Class which produces validation between the simulated data and the real data
        :param sim_root: root path of the simulated data
        :param real_root: root path of the real data
        :param start_time: datetime which
        :param sampling_window: number of seconds before start_time to sample from
        :param simulation_window: number of seconds after start_time to simulate
        :param orderbook_window: number of seconds before start_time to analyse to get state of the orderbook
        """
        self.logger = logging.getLogger()

        self.real_root = real_root
        self.sim_root = sim_root

        self.start_time = start_time
        self.sampling_window = sampling_window
        self.simulation_window = simulation_window
        self.orderbook_window = orderbook_window

        self.product = product
        self.params_path = params_path

        self.sampling_window_start_time = self.start_time - timedelta(seconds=self.sampling_window)
        self.sampling_window_end_time = self.start_time

        self.orderbook_window_start_time = start_time - datetime.timedelta(seconds=self.orderbook_window)
        self.orderbook_window_end_time = start_time

    def run_simulation(self):
        # Get parameters
        orders_df, trades_df, cancels_df = DataLoader.load_sampling_data(self.real_root, self.sampling_window_start_time,
                                                                         self.sampling_window_end_time, self.product)
        real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined BTC-USD")
        params = real_analysis.generate_order_params()
        real_analysis.params_to_file(params, self.params_path)

        self.logger.info("Parameters saved to: " + self.params_path)

        # Get orderbook
        orders_df, trades_df, cancels_df = DataLoader.load_sampling_data(self.real_root, self.orderbook_window_start_time,
                                                                         self.orderbook_window_end_time, self.product)

        orderbook = OrderBook.orderbook_from_df(orders_df, trades_df, cancels_df)
        # TODO: remove this magic string
        orderbook_path = "/Users/jamesprince/project-data/orderbook-" + self.orderbook_window_end_time.isoformat() + ".csv"
        OrderBook.orderbook_to_file(orderbook, orderbook_path)

        self.logger.info("Orderbook saved to: " + orderbook_path)

        # Generate .conf file
        # TODO: generate this in a function further up the chain
        config = {'paths': {
            'simRoot': "/Users/jamesprince/project-data/sims/",
            'params': "/Users/jamesprince/project-data/parameters.json",
            'orderbook': orderbook_path
            },

            'execution': {
                'numSimulations': 10,
                'parallel': True,
                'logLevel': "INFO"
            },

            'orderbook': {
                'stp': False
            }}
        config_string = SimConfig.generate_config_string(config)
        config_path = "/Users/jamesprince/project-data/analysis.conf"
        f = open(config_path, 'w')
        f.write(config_string)
        f.close()

        self.logger.info("Wrote sim config to: " + config_path)

        # Start simulation
        # TODO: remove hard path
        jar_path = '/Users/jamesprince/project-data/orderbooksimulator_jar/orderbooksimulator.jar'
        bash_command = "java -jar " + jar_path + " " + config_path

        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        self.logger.info("Running simulations")
        print(str(output))

        process.wait()
        self.logger.info("Simulations complete")

        # Validate
        sim_analysis = SimulationAnalysis(self.sim_root, "Combined Analysis")
        self.graph_real_prices_with_simulated_confidence_intervals(sim_analysis)

    def graph_real_prices_with_simulated_confidence_intervals(self, sim_analysis: SimulationAnalysis):
        interval = 10  # seconds
        times = list(range(interval, self.simulation_window, interval))

        confidence_intervals = sim_analysis.calculate_confidence_at_times(times)
        print(confidence_intervals)
        self.logger.debug(confidence_intervals)

        real_times, real_prices = self.__fetch_real_data()

        start_price = real_prices.iloc[0]

        # Add the initial price and then convert to numpy arrays
        mean0 = np.array([start_price] + list(map(lambda lu: (lu[0] + lu[1]) / 2, confidence_intervals.values())))
        ub0 = np.array([start_price] + list(map(lambda x: x[1], confidence_intervals.values())))
        lb0 = np.array([start_price] + list(map(lambda x: x[0], confidence_intervals.values())))

        times = [0] + times

        # plot the data
        self.__plot_mean_and_ci_and_real_values(mean0, ub0, lb0, times, real_times, real_prices, color_mean='k',
                                                color_shading='k')

        plt.show()

    def print_stat_comparison(self):
        real_orders, _, _ = DataLoader.load_sampling_data(self.real_root, self.sampling_window_start_time,
                                                          self.sampling_window_end_time, self.product)
        sim_orders = self.sim_analysis.all_sims[0][0].compute()

        real_stats = Statistics.get_order_stats(real_orders)
        sim_stats = Statistics.get_order_stats(sim_orders)

        print_str = ""

        for k in real_stats.keys():
            print_str += k + "\t\t\t\tReal: " + str(real_stats[k]) + "\t\t\tSim: " + str(sim_stats[k]) + "\n"

        print(print_str)

    # Source: https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/
    def __plot_mean_and_ci_and_real_values(self, mean, lb, ub, times, real_times, real_prices, color_mean=None,
                                           color_shading=None):
        # plot the shaded range of the confidence intervals
        plt.figure(figsize=(12, 8))
        # plt.ylim(8000, 9000)
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
        df = DataLoader().load_real_data(self.real_root, self.start_time,
                                         self.start_time + timedelta(seconds=self.simulation_window), self.product)
        # [['time', 'price', 'reason']]
        trades_df = DataSplitter.get_trades(df)

        print(trades_df)

        GraphCreator("BTC").graph_price_time(trades_df, "BTC")

        trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
        real_times = trades_df['time']
        real_prices = trades_df['price']
        return real_times, real_prices
