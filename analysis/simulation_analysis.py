import csv
import datetime
import logging
import math
import os
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from data_loader import DataLoader
from data_utils import DataUtils
from graph_creator import GraphCreator
from stats import Statistics


class SimulationAnalysis:
    logger = logging.getLogger()

    def __init__(self, config, sim_st: datetime.datetime):
        """
        :param config: initial application config
        """
        self.logger = logging.getLogger()

        self.sim_root = config.sim_root \
                        + sim_st.date().isoformat() + "/" \
                        + sim_st.time().isoformat() + "/"

        confidence_dir = config.confidence_root \
                         + sim_st.date().isoformat() + "/"

        pathlib.Path(confidence_dir).mkdir(parents=True, exist_ok=True)
        self.confidence_path = confidence_dir + sim_st.time().isoformat() + ".dump"

        self.graph_creator = GraphCreator(config, "Simulation " + config.product)

        self.all_sims = DataLoader().load_sim_data(self.sim_root, 0, config.num_simulators)

    def analyse(self):
        # logger.debug(self.dirs)
        # for directory in self.dirs:
        i = 0
        for orders_dd, trades_dd, cancels_dd in self.all_sims:
            orders_df = orders_dd.compute()
            trades_df = trades_dd.compute()
            # self.graph_creator.graph_order_sizes(orders_df)
            # self.graph_creator.graph_price_quantity(orders_df)
            mid = trades_df['price'].dropna().iloc[0]
            self.graph_creator.graph_price_time(orders_df, "orders (sim #" + str(i) + ")", mid, self.ywindow)
            # self.graph_creator.graph_time_delta(orders_df)
            #
            # self.graph_creator.graph_price_time(trades_df, "trades (sim #" + str(i) + ")")
            #
            # # self.graph_creator.graph_relative_price_distribution(trades_df, cancels_df, 20)
            #
            # self.graph_creator.graph_relative_price_distribution(trades_df, orders_df, 20)
            # self.graph_creator.graph_interval(orders_df)

            # graphs.graph_price_quantity(trades_df)
            plt.show()
            i += 1

    def dump_confidence_data(self, dst, time_prices_dict: dict, time_confidence_dict: dict):
        self.logger.info("Dumping confidence data to " + dst)
        with open(dst, 'w', newline='\n') as fd:
            fd.write(str(time_prices_dict))
            fd.write(str(time_confidence_dict))

        self.logger.info("Confidence data dumped to: " + dst)

    # TODO: fix some of these awful names, such as "seconds"
    def calculate_confidence_at_times(self, seconds_list: List[int], level=0.95):
        time_prices_dict = self.extract_prices_at_times(self.all_sims, seconds_list)
        time_confidence_dict = self.calculate_confidences(time_prices_dict, level)

        self.dump_confidence_data(self.confidence_path, time_prices_dict, time_confidence_dict)

        return time_confidence_dict

    def extract_prices_at_times(self, all_sims, seconds_list):
        time_prices_dict = {}
        for seconds in seconds_list:
            time_prices_dict[seconds] = []
        sim_index = 0
        for sim in all_sims:
            _, trades_dd, _ = sim
            trades_df = trades_dd.compute()
            for seconds in seconds_list:
                price = DataUtils.get_last_price_before(trades_df, seconds)
                time_prices_dict[seconds].append(price)
            sim_index += 1
        return time_prices_dict

    @classmethod
    def calculate_confidences(cls, time_prices_dict, level: float):
        time_confidence_dict = {}
        num_nans = {}
        low_high = {}
        for time_in_seconds, prices in time_prices_dict.items():
            filt_prices = list(filter(lambda p: not math.isnan(p), prices))
            low_high[time_in_seconds] = (min(filt_prices), max(filt_prices))
            num_nans[time_in_seconds] = len(prices) - len(filt_prices)
            interval = st.t.interval(level, len(filt_prices) - 1, loc=np.mean(filt_prices), scale=st.sem(filt_prices))
            time_confidence_dict[time_in_seconds] = interval

        for key in num_nans:
            cls.logger.info(str(key) + " has " + str(num_nans[key]) + " NaNs")

        for key, lh in low_high.items():
            cls.logger.info(str(key) + " low, high: " + str(lh))

        for key, intervals in time_confidence_dict.items():
            cls.logger.info(str(key) + " has confidence width: " + str(abs(intervals[0] - intervals[1])))

        return time_confidence_dict
