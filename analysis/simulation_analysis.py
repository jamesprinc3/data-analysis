import logging
import math
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from data_loader import DataLoader
from data_utils import DataUtils
from graph_creator import GraphCreator
from stats import Statistics


class SimulationAnalysis:
    def __init__(self, config):
        """
        :param config: initial application config
        """
        self.logger = logging.getLogger()

        root_path = config['full_paths']['root']
        self.sim_root = root_path + config['part_paths']['sim_root']
        self.graph_creator = GraphCreator("Simulation " + config['data']['product'])

        self.all_sims = DataLoader().load_sim_data(self.sim_root, 0, 100)

    def analyse(self):
        # logger.debug(self.dirs)
        # for directory in self.dirs:
        orders_dd, trades_dd, cancels_dd = DataLoader().load_sim_data(self.sim_root)[0]

        orders_df = orders_dd.compute()
        # self.graph_creator.graph_order_sizes(orders_df)
        # self.graph_creator.graph_price_quantity(orders_df)
        self.graph_creator.graph_price_time(orders_df, "orders")
        # self.graph_creator.graph_time_delta(orders_df)
        #
        trades_df = trades_dd.compute()
        self.graph_creator.graph_price_time(trades_df, "trades")

        # self.graph_creator.graph_relative_price_distribution(trades_df, cancels_df, 20)

        self.graph_creator.graph_relative_price_distribution(trades_df, orders_df, 20)
        self.graph_creator.graph_interval(orders_df)

        # graphs.graph_price_quantity(trades_df)

        plt.show()

    # TODO: fix some of these awful names, such as "seconds"
    def calculate_confidence_at_times(self, seconds_list: List[int], level=0.95):
        time_prices_dict = self.extract_prices_at_times(self.all_sims, seconds_list)
        self.logger.debug(time_prices_dict)
        time_confidence_dict = self.calculate_confidences(time_prices_dict, level)

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

    @staticmethod
    def calculate_confidences(time_prices_dict, level: float):
        time_confidence_dict = {}
        for time_in_seconds, prices in time_prices_dict.items():
            filt_prices = list(filter(lambda p: not math.isnan(p), prices))
            interval = st.t.interval(level, len(filt_prices) - 1, loc=np.mean(filt_prices), scale=st.sem(filt_prices))
            time_confidence_dict[time_in_seconds] = interval

        return time_confidence_dict



