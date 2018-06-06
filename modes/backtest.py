import csv
import datetime
import logging
import os
import pathlib
import signal
import subprocess
import time
from datetime import timedelta

import numpy as np
from pebble import concurrent

from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from data.data_utils import DataUtils
from modes.sample import Sample
from modes.simulation_analysis import SimulationAnalysis
from orderbook import OrderBook, reconstruct_orderbook
from output.graphing import Graphing
from output.writer import Writer
from sim_config import SimConfig
from stats import Statistics


class Backtest:
    def __init__(self, config, sim_st: datetime, all_ob_data, all_sampling_data, all_future_data):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config

        self.sim_root = self.config.sim_root + sim_st.date().isoformat() + "/" + sim_st.time().isoformat() + "/"

        self.params_root = self.config.params_root + sim_st.date().isoformat() + "/"
        pathlib.Path(self.params_root).mkdir(parents=True, exist_ok=True)

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

    def prepare_simulation(self):
        try:
            params_path = self.params_root \
                          + self.sim_st.time().isoformat() + ".json"
            if self.config.use_cached_params and os.path.isfile(params_path):
                self.logger.info("Params file exists, therefore we're using it! " + params_path)
            else:
                self.logger.info("Not using params cache" + "\nGenerating params...")
                # Get parameters
                orders_df, trades_df, cancels_df = self.all_sampling_data
                params = Sample.generate_sim_params(orders_df, trades_df, cancels_df)

                # Save params (that can be reused!)
                Writer.json_to_file(params, params_path)
                self.logger.info("Permanent parameters saved to: " + params_path)

            ob_final = reconstruct_orderbook(self.all_ob_data, self.config, self.sim_st, self.logger)

            # Save orderbook
            orderbook_path = self.config.orderbook_output_root + self.orderbook_window_end_time.isoformat() + ".csv"
            OrderBook.orderbook_to_file(ob_final, orderbook_path)

            # Generate .conf file
            sim_config = self.generate_config_dict(orderbook_path, params_path)
            sim_config_string = SimConfig.generate_config_string(sim_config)
            self.save_config(sim_config_string)
            return True
        except Exception as e:
            self.logger.error(
                "Simulation preparation failed, skipping, at: " + self.sim_st.isoformat() + "\nError was\n" + str(e))
            return False

    @concurrent.process(timeout=300)
    def run_simulation(self):
        self.logger.info("Running simulations")
        bash_command = "java -jar " + self.config.jar_path + " " + self.config.sim_config_path
        # Have to initialise name so it can be terminated in the finally clause
        sim_process = None
        try:
            t0 = time.time()
            sim_process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, preexec_fn=os.setsid)
            output, error = sim_process.communicate(timeout=self.config.sim_timeout)

            self.logger.info("Simulations complete, took " + str(time.time() - t0) + " seconds")

            with open(self.sim_logs_path, 'w') as f:
                f.write(str(output))
                f.write("\n")
                f.write(str(error))

            self.logger.info("Writing output and error to: " + self.sim_logs_path)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(sim_process.pid), signal.SIGTERM)
            self.logger.error("Timeout limit for sim was reached, JVM killed prematurely.")

    def save_config(self, sim_config_string):
        try:
            f = open(self.config.sim_config_path, 'w')
            f.write(sim_config_string)
            f.close()
            self.logger.info("Wrote sim config to: " + self.config.sim_config_path)
        except Exception as e:
            self.logger.error("Failed writing sim config to " + self.config.sim_config_path + " exception: " + str(e))

    def generate_config_dict(self, orderbook_path, params_path):
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
        return sim_config

    def validate_analyses(self, prog_start: datetime.datetime):
        sim_analysis = SimulationAnalysis(self.config, self.sim_st)
        trade_price_validation_data = self.get_trade_validation_data(sim_analysis)
        mid_price_validation_data = self.get_midprice_validation_data(sim_analysis)
        monte_carlo_data = self.get_monte_carlo_data(sim_analysis)

        if self.config.spread:
            best_bid_data = self.get_bid_data(sim_analysis)
            best_ask_data = self.get_ask_data(sim_analysis)
            self.graphing.plot_spread(best_bid_data, best_ask_data)

        trade_correlation_file_path = self.config.correlation_root + prog_start.isoformat() + "-trade.csv"
        self.append_final_prices(trade_correlation_file_path, trade_price_validation_data[0],
                                 trade_price_validation_data[1],
                                 trade_price_validation_data[2], trade_price_validation_data[5])
        midprice_correlation_file_path = self.config.correlation_root + prog_start.isoformat() + "-midprice.csv"
        self.append_final_prices(midprice_correlation_file_path, mid_price_validation_data[0],
                                 mid_price_validation_data[1],
                                 mid_price_validation_data[2], mid_price_validation_data[5])

        self.graph_trade_prices(*trade_price_validation_data)
        self.graph_mid_prices(*mid_price_validation_data)
        self.graph_monte_carlo_data(trade_price_validation_data[5].iloc[0], monte_carlo_data)

    def append_final_prices(self, dst, sim_means, sim_ubs, sim_lbs, real_prices):
        start_price = real_prices.dropna().iloc[0]

        last_real_price = real_prices.dropna().iloc[-1]
        last_sim_price_mean = sim_means[-1]
        last_sim_price_ub = sim_ubs[-1]
        last_sim_price_lb = sim_lbs[-1]

        if not os.path.isfile(dst):
            with open(dst, 'w', newline='') as csvfile:
                fieldnames = ['start_time', 'start_price', 'last_real_price',
                              'last_sim_price_mean', 'last_sim_price_ub', 'last_sim_price_lb']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()

        with open(dst, 'a', newline='') as fd:
            row = ",".join(
                [self.sim_st.isoformat(), str(start_price), str(last_real_price),
                 str(last_sim_price_mean), str(last_sim_price_ub), str(last_sim_price_lb)
                 ]) + "\n"
            fd.write(row)

    def get_trade_validation_data(self, sim_analysis: SimulationAnalysis):
        times = self.get_times()

        confidence_intervals = sim_analysis.calculate_trade_confidence_at_times(times)
        self.logger.info(confidence_intervals)

        real_times, real_prices = self.__fetch_real_data()

        start_price = real_prices.iloc[0]

        # Add the initial price and then convert to numpy arrays
        sim_means = np.array([start_price] + list(map(lambda lu: (lu[0] + lu[1]) / 2, confidence_intervals.values())))
        sim_ub = np.array([start_price] + list(map(lambda x: x[1], confidence_intervals.values())))
        sim_lb = np.array([start_price] + list(map(lambda x: x[0], confidence_intervals.values())))
        times = [0] + times

        return sim_means, sim_ub, sim_lb, times, real_times, real_prices

    #TODO: reduce duplication
    def get_midprice_validation_data(self, sim_analysis: SimulationAnalysis):
        times = self.get_times()

        time_prices_dict = sim_analysis.extract_mid_prices_at_times(sim_analysis.all_sims, times)

        percentiles = sim_analysis.calculate_midprice_percentiles(time_prices_dict)
        self.logger.info(percentiles)

        real_times, real_prices = self.__fetch_real_data()

        start_price = real_prices.iloc[0]

        # Add the initial price and then convert to numpy arrays
        sim_medians = np.array([start_price] + list(map(lambda lmu: lmu[1], percentiles.values())))
        sim_upper = np.array([start_price] + list(map(lambda lmu: lmu[2], percentiles.values())))
        sim_lower = np.array([start_price] + list(map(lambda lmu: lmu[0], percentiles.values())))
        times = [0] + times

        return sim_medians, sim_upper, sim_lower, times, real_times, real_prices

    def get_times(self):
        times = list(range(self.config.xinterval,
                           self.config.simulation_window + self.config.xinterval,
                           self.config.xinterval))
        return times

    def graph_trade_prices(self, sim_means, sim_ub, sim_lb, times, real_times,
                           real_prices):
        self.config.plt.title(self.config.product + " at " + self.__get_plot_title())
        self.config.plt.xlabel("Time (seconds)")
        self.config.plt.ylabel("Price ($)")

        # plot the data
        self.graphing.plot_mean_and_ci_and_real_values(sim_means, sim_ub, sim_lb, times, real_times, real_prices,
                                                       color_mean='k',
                                                       color_shading='k')
        self.output_graph("tradeprices", self.__get_plot_title())

        self.config.plt.close()

    def output_graph(self, category, title):
        self.config.plt.title(title + " " + category)

        if self.config.graph_mode == "save":
            plot_root = self.config.graphs_root + category + "/" + self.sim_st.date().isoformat() + "/"
            self.save_figure(plot_root, title)
        if self.config.graph_mode == "show":
            self.config.plt.show()

    def save_figure(self, plot_root, title):
        # Ensure directory exists
        pathlib.Path(plot_root).mkdir(parents=True, exist_ok=True)

        plot_path = plot_root + title + ".png"

        self.config.plt.savefig(plot_path, dpi=600, transparent=True)
        self.logger.info("Saved plot to: " + plot_path)

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

    def get_monte_carlo_data(self, sim_analysis):
        sim_prices = {}
        times = self.get_times()

        for index in range(0, len(sim_analysis.all_sims)):
            sim_prices[index] = []

        for index in range(0, len(sim_analysis.all_sims)):
            sim = sim_analysis.all_sims[index]
            trades_dd = sim[1]
            trades_df = trades_dd.compute()
            if len(trades_df) == 0:
                continue
            trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
            for seconds in times:
                price = DataUtils.get_last_price_before(trades_df, seconds)
                sim_prices[index].append(price)
        return sim_prices

    def graph_monte_carlo_data(self, start_price, monte_carlo_data):

        self.config.plt.figure(figsize=(12, 8))
        ymin = start_price - (self.config.ywindow / 2)
        ymax = start_price + (self.config.ywindow / 2)
        self.config.plt.ylim(ymin, ymax)

        times = self.get_times()

        for _, prices_for_sim in monte_carlo_data.items():
            if len(prices_for_sim) == 0:
                continue
            self.config.plt.plot(times, prices_for_sim)

        self.output_graph("monte", self.__get_plot_title())

    def graph_mid_prices(self, sim_mid_means, sim_mid_ub, sim_mid_lb, times, real_times,
                         real_prices):
        self.config.plt.title(self.config.product + " at " + self.__get_plot_title())
        self.config.plt.xlabel("Time (seconds)")
        self.config.plt.ylabel("Price ($)")

        # plot the data
        self.graphing.plot_mean_and_ci_and_real_values(sim_mid_means, sim_mid_ub, sim_mid_lb, times, real_times, real_prices,
                                                       color_mean='k',
                                                       color_shading='k')
        self.output_graph("midprices", self.__get_plot_title())

        self.config.plt.close()

    def get_bid_data(self, sim_analysis):
        ret = {}

        for index in range(0, len(sim_analysis.all_sims)):
            sim = sim_analysis.all_sims[index]
            best_bids_dd = sim[4]
            best_bids_df = best_bids_dd.compute()
            if len(best_bids_df) == 0:
                continue
            best_bids_df['time'] = DataUtils().get_times_in_seconds_after_start(best_bids_df['time'])
            ret[index] = best_bids_df
        return ret

    def get_ask_data(self, sim_analysis):
        ret = {}

        for index in range(0, len(sim_analysis.all_sims)):
            sim = sim_analysis.all_sims[index]
            _, _, _, _, _, best_asks_dd = sim
            best_bids_df = best_asks_dd.compute()
            if len(best_bids_df) == 0:
                continue
            best_bids_df['time'] = DataUtils().get_times_in_seconds_after_start(best_bids_df['time'])
            ret[index] = best_bids_df
        return ret
