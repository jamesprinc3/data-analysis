import csv
import datetime
import logging
import os
import signal
import subprocess
from datetime import timedelta

from pebble import concurrent

import numpy as np
import pathlib

from modes.real_analysis import RealAnalysis
from modes.sample import Sample
from modes.simulation_analysis import SimulationAnalysis
from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from data.data_utils import DataUtils
from orderbook import OrderBook
from output.graphing import Graphing
from sim_config import SimConfig
from stats import Statistics
from output.writer import Writer


class Backtest:
    def __init__(self, config, sim_st: datetime, all_ob_data, all_sampling_data, all_future_data):
        self.logger = logging.getLogger()

        self.config = config

        self.sim_root = self.config.sim_root + sim_st.date().isoformat() + "/" + sim_st.time().isoformat() + "/"

        self.params_path = self.config.params_path + sim_st.date().isoformat() + "/"
        pathlib.Path(self.params_path).mkdir(parents=True, exist_ok=True)

        sim_logs_dir = self.config.sim_logs_root + sim_st.date().isoformat() + "/"

        pathlib.Path(sim_logs_dir).mkdir(parents=True, exist_ok=True)

        self.sim_logs_path = sim_logs_dir + sim_st.time().isoformat() + ".log"

        self.sim_st = sim_st

        self.sampling_window_start_time = sim_st - timedelta(seconds=self.config.sampling_window)
        self.sampling_window_end_time = sim_st

        self.orderbook_window_start_time = sim_st - datetime.timedelta(seconds=self.config.orderbook_window)
        self.orderbook_window_end_time = sim_st

        self.all_sampling_data = all_sampling_data
        self.all_ob_data = all_ob_data
        self.all_future_data = all_future_data

        self.graphing = Graphing(config, "Backtest @ " + sim_st.isoformat())

        import matplotlib

        matplotlib.use('PS')

        import matplotlib.pyplot as plt

        self.plt = plt

    def run_simulation(self):
        params_path = self.params_path \
                      + self.sim_st.time().isoformat() + ".json"
        if self.config.use_cached_params and os.path.isfile(params_path):
            self.logger.info("Params file exists, therefore we're using it! " + params_path)
        else:
            self.logger.info("Not using params cache" + "\nGenerating params...")
            # Get parameters
            orders_df, trades_df, cancels_df = self.all_sampling_data
            real_analysis = RealAnalysis("Backtest BTC-USD")
            params = Sample.generate_order_params(trades_df, orders_df, cancels_df)

            # Save params (that can be reused!)
            Writer.json_to_file(params, params_path)
            self.logger.info("Permanent parameters saved to: " + params_path)

        # Get orderbook
        orders_df, trades_df, cancels_df = self.all_ob_data
        closest_state_file_path = OrderBook.locate_closest_ob_state(self.config.orderbook_input_root, self.sim_st)
        ob_state_df = OrderBook().load_orderbook_state(closest_state_file_path)
        ob_final = OrderBook().get_orderbook(orders_df, trades_df, cancels_df, ob_state_df)

        # Save orderbook
        orderbook_path = self.config.orderbook_root + self.orderbook_window_end_time.isoformat() + ".csv"
        OrderBook.orderbook_to_file(ob_final, orderbook_path)
        self.logger.info("Orderbook saved to: " + orderbook_path)

        # Generate .conf file
        # TODO: generate this in a function further up the chain
        sim_config = {'paths': {
            'simRoot': self.sim_root,
            'params': params_path,
            'orderbook': orderbook_path
        },

            'execution': {
                'numSimulations': self.config.num_simulators,
                'simulationSeconds': self.config.simulation_window,
                'numTraders': self.config.num_traders,
                'parallel': True,
                'logLevel': "INFO"
            },

            'orderbook': {
                'stp': False
            }}
        sim_config_string = SimConfig.generate_config_string(sim_config)

        f = open(self.config.sim_config_path, 'w')
        f.write(sim_config_string)
        f.close()

        self.logger.info("Wrote sim config to: " + self.config.sim_config_path)

        # Start simulation
        bash_command = "java -jar " + self.config.jar_path + " " + self.config.sim_config_path

        self.logger.info("Running simulations")

        # Have to initialise name so it can be terminated in the finally clause
        sim_process = None

        try:
            sim_process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, preexec_fn=os.setsid)
            output, error = sim_process.communicate(timeout=self.config.sim_timeout)

            self.logger.info("Simulations complete")

            with open(self.sim_logs_path, 'w') as f:
                f.write(str(output))
                f.write("\n")
                f.write(str(error))

            self.logger.info("Writing output and error to: " + self.sim_logs_path)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(sim_process.pid), signal.SIGTERM)
            self.logger.error("Timeout limit for sim was reached, JVM killed prematurely.")

    @concurrent.process(timeout=None)
    def validate_analyses(self, prog_start: datetime.datetime):
        sim_analysis = SimulationAnalysis(self.config, self.sim_st)
        validation_data = self.get_validation_data(sim_analysis)

        correlation_file_path = self.config.correlation_root + prog_start.isoformat() + ".csv"
        self.append_final_prices(correlation_file_path, validation_data[0], validation_data[5])
        self.graph_real_prices_with_simulated_confidence_intervals(*validation_data)

    def append_final_prices(self, dst, sim_means, real_prices):
        start_price = real_prices.dropna().iloc[0]

        last_real_price = real_prices.dropna().iloc[-1]
        last_sim_price = sim_means[-1]

        if not os.path.isfile(dst):
            with open(dst, 'w', newline='') as csvfile:
                fieldnames = ['start_time', 'start_price', 'last_real_price', 'last_sim_price']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

        with open(dst, 'a', newline='') as fd:
            row = ",".join(
                [self.sim_st.isoformat(), str(start_price), str(last_real_price), str(last_sim_price)]) + "\n"
            fd.write(row)

    def get_validation_data(self, sim_analysis: SimulationAnalysis):
        times = list(range(self.config.xinterval,
                           self.config.simulation_window + self.config.xinterval,
                           self.config.xinterval))

        confidence_intervals = sim_analysis.calculate_confidence_at_times(times)
        self.logger.info(confidence_intervals)

        real_times, real_prices = self.__fetch_real_data()

        start_price = real_prices.iloc[0]

        # Add the initial price and then convert to numpy arrays
        sim_means = np.array([start_price] + list(map(lambda lu: (lu[0] + lu[1]) / 2, confidence_intervals.values())))
        sim_ub = np.array([start_price] + list(map(lambda x: x[1], confidence_intervals.values())))
        sim_lb = np.array([start_price] + list(map(lambda x: x[0], confidence_intervals.values())))
        times = [0] + times

        return sim_means, sim_ub, sim_lb, times, real_times, real_prices

    def graph_real_prices_with_simulated_confidence_intervals(self, sim_means, sim_ub, sim_lb, times, real_times,
                                                              real_prices):
        self.plt.title(self.config.product + " at " + self.__get_plot_title())
        self.plt.xlabel("Time (seconds)")
        self.plt.ylabel("Price ($)")

        # plot the data
        self.graphing.plot_mean_and_ci_and_real_values(sim_means, sim_ub, sim_lb, times, real_times, real_prices,
                                                       color_mean='k',
                                                       color_shading='k')

        if self.config.save_graphs:
            plot_root = self.config.graphs_root + self.sim_st.date().isoformat()
            # Ensure directory exists
            pathlib.Path(plot_root).mkdir(parents=True, exist_ok=True)

            # Save plot
            plot_path = plot_root + self.__get_plot_title() + ".png"
            self.plt.savefig(plot_path, dpi=600, transparent=True)
            self.logger.info("Saved plot to: " + plot_path)

        if self.config.show_graphs:
            self.plt.show()

        self.plt.close()

    def __get_plot_title(self):
        plot_path = self.sim_st.time().isoformat() \
                    + "-sample-" + str(self.config.sampling_window) \
                    + "-sim_window-" + str(self.config.simulation_window) \
                    + "-num_sims-" + str(self.config.num_simulators)
        return plot_path

    def print_stat_comparison(self):
        real_orders, _, _ = DataLoader.load_split_data(self.config.real_root, self.sampling_window_start_time,
                                                       self.sampling_window_end_time, self.config.product)
        sim_orders = self.sim_analysis.all_sims[0][0].compute()

        real_stats = Statistics.get_order_stats(real_orders)
        sim_stats = Statistics.get_order_stats(sim_orders)

        print_str = ""

        for k in real_stats.keys():
            print_str += k + "\t\t\t\tReal: " + str(real_stats[k]) + "\t\t\tSim: " + str(sim_stats[k]) + "\n"

        print(print_str)

    def __fetch_real_data(self):
        df = DataLoader().load_feed(self.config.real_root, self.sim_st,
                                    self.sim_st + timedelta(seconds=self.config.simulation_window), self.config.product)
        # [['time', 'price', 'reason']]
        trades_df = DataSplitter.get_trades(df)

        trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
        trades_df['price'].iloc[0] = DataUtils().get_first_non_nan(trades_df['price'])
        real_times = trades_df['time']
        real_prices = trades_df['price']
        return real_times, real_prices
