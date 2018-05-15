import datetime
import logging
import subprocess
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pathlib

from analysis.real_analysis import RealAnalysis
from analysis.simulation_analysis import SimulationAnalysis
from data_loader import DataLoader
from data_splitter import DataSplitter
from data_utils import DataUtils
from orderbook import OrderBook
from sim_config import SimConfig
from stats import Statistics


class CombinedAnalysis:
    def __init__(self, config, sim_st: datetime, all_ob_data, all_sampling_data, all_future_data):
        self.logger = logging.getLogger()

        self.config = config

        self.real_root = config['paths']['real_root']
        self.sim_root = config['paths']['sim_root']
        self.graphs_root = config['paths']['graphs_root']
        self.params_root = config['paths']['params_root']
        self.orderbook_root = config['paths']['orderbook_root']

        self.temp_params_path = config['paths']['params_path']
        self.sim_config_path = config['paths']['sim_config_path']
        self.jar_path = config['paths']['jar_path']

        self.sampling_window = int(config['window']['sampling'])
        self.simulation_window = int(config['window']['simulation'])
        self.orderbook_window = int(config['window']['orderbook'])

        self.product = config['data']['product']
        self.sim_st = sim_st

        self.mode = config['behaviour']['mode']
        self.show_graphs = config['behaviour'].getboolean('show_graphs')
        self.save_graphs = config['behaviour'].getboolean('save_graphs')
        self.fit_distributions = config['behaviour'].getboolean('fit_distributions')
        self.sim_timeout = int(config['behaviour']['sim_timeout'])
        self.num_simulators = int(config['behaviour']['num_simulators'])

        self.ywindow = int(config['graphs']['ywindow'])

        self.sampling_window_start_time = sim_st - timedelta(seconds=self.sampling_window)
        self.sampling_window_end_time = sim_st

        self.orderbook_window_start_time = sim_st - datetime.timedelta(seconds=self.orderbook_window)
        self.orderbook_window_end_time = sim_st

        self.all_sampling_data = all_sampling_data
        self.all_ob_data = all_ob_data
        self.all_future_data = all_future_data

    def run_simulation(self):
        # Get parameters
        orders_df, trades_df, cancels_df = self.all_sampling_data
        real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined BTC-USD")
        params = real_analysis.generate_order_params()

        # Save temporary params (that the simulator will use)
        real_analysis.params_to_file(params, self.temp_params_path)
        self.logger.info("Temporary parameters saved to: " + self.temp_params_path)

        # Save permanent params (that can be reused!)
        perma_params_path = self.params_root + self.sim_st.date().isoformat() + "/"
        pathlib.Path(perma_params_path).mkdir(parents=True, exist_ok=True)
        perma_params_path = perma_params_path + self.sim_st.time().isoformat() + ".json"
        real_analysis.params_to_file(params, perma_params_path)
        self.logger.info("Permanent parameters saved to: " + self.temp_params_path)

        # Get orderbook
        orders_df, trades_df, cancels_df = self.all_ob_data
        orderbook = OrderBook.orderbook_from_df(orders_df, trades_df, cancels_df)
        # TODO: remove this magic string
        orderbook_path = self.orderbook_root + self.orderbook_window_end_time.isoformat() + ".csv"
        OrderBook.orderbook_to_file(orderbook, orderbook_path)

        self.logger.info("Orderbook saved to: " + orderbook_path)

        # Generate .conf file
        # TODO: generate this in a function further up the chain
        sim_config = {'paths': {
            'simRoot': self.sim_root,
            'params': self.temp_params_path,
            'orderbook': orderbook_path
        },

            'execution': {
                'numSimulations': self.num_simulators,
                'parallel': True,
                'logLevel': "INFO"
            },

            'orderbook': {
                'stp': False
            }}
        sim_config_string = SimConfig.generate_config_string(sim_config)

        f = open(self.sim_config_path, 'w')
        f.write(sim_config_string)
        f.close()

        self.logger.info("Wrote sim config to: " + self.sim_config_path)

        # Start simulation

        bash_command = "java -jar " + self.jar_path + " " + self.sim_config_path

        self.logger.info("Running simulations")

        try:
            process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate(timeout=self.sim_timeout)

            self.logger.info("output: " + str(output) + " error: " + str(error))
            self.logger.info("Simulations complete")

            # Validate
            sim_analysis = SimulationAnalysis(self.config)
            self.graph_real_prices_with_simulated_confidence_intervals(sim_analysis)
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout limit for sim was reached, JVM killed.")

    def graph_real_prices_with_simulated_confidence_intervals(self, sim_analysis: SimulationAnalysis):
        interval = 10  # seconds
        times = list(range(interval, self.simulation_window, interval))

        confidence_intervals = sim_analysis.calculate_confidence_at_times(times)
        self.logger.info(confidence_intervals)

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

        if self.save_graphs:
            plot_root = self.graphs_root + self.sim_st.date().isoformat()
            # Ensure directory exists
            pathlib.Path(plot_root).mkdir(parents=True, exist_ok=True)

            # Save plot
            plot_path = plot_root \
                        + "/" + self.sim_st.time().isoformat() \
                        + "-sample-" + str(self.sampling_window) \
                        + "-sim_window-" + str(self.simulation_window) \
                        + "-num_sims-" + str(self.num_simulators) \
                        + ".png"
            plt.savefig(plot_path, dpi=600, transparent=True)
            self.logger.info("Saved plot to: " + plot_path)

        if self.show_graphs:
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
        # Set bounds and make title (+ for axes)
        plt.figure(figsize=(12, 8))
        ymin = real_prices.iloc[0] - (self.ywindow / 2)
        ymax = real_prices.iloc[0] + (self.ywindow / 2)
        plt.ylim(ymin, ymax)
        plt.title(self.product + " at " + self.sim_st.isoformat()
                  + " sampling_window=" + str(self.sampling_window)
                  + " simulation_window=" + str(self.simulation_window)
                  + " num_simulators=" + str(self.num_simulators))
        plt.xlabel("Time (seconds)")
        plt.ylabel("Price ($)")

        # plot the shaded range of the confidence intervals
        plt.fill_between(times, ub, lb, color=color_shading, alpha=.5,
                         label="Simulated 95% Confidence Interval")

        # plot the mean on top
        plt.plot(times, mean, color_mean,
                 label="Simulated Mean")
        plt.plot(real_times, real_prices, 'r+',
                 label="Real Trades")
        plt.legend(loc='upper right')

    def __fetch_real_data(self):
        df = DataLoader().load_real_data(self.real_root, self.sim_st,
                                         self.sim_st + timedelta(seconds=self.simulation_window), self.product)
        # [['time', 'price', 'reason']]
        trades_df = DataSplitter.get_trades(df)

        trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
        trades_df['price'].iloc[0] = DataUtils().get_first_non_nan(trades_df['price'])
        real_times = trades_df['time']
        real_prices = trades_df['price']
        return real_times, real_prices
