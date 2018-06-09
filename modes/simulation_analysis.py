import datetime
import logging
import math
import pathlib
from typing import List

import pandas as pd

from data.data_loader import DataLoader
from data.data_utils import DataUtils
from output.graphing import Graphing


class SimulationAnalysis:
    logger = logging.getLogger("SimulationAnalysis")

    def __init__(self, config, sim_st: datetime.datetime):
        """
        :param config: initial application config
        """
        self.config = config

        self.sim_root = config.sim_root \
                        + sim_st.date().isoformat() + "/" \
                        + sim_st.time().isoformat() + "/"

        confidence_dir = config.confidence_root \
                         + sim_st.date().isoformat() + "/"

        pathlib.Path(confidence_dir).mkdir(parents=True, exist_ok=True)
        self.confidence_path = confidence_dir + sim_st.time().isoformat() + ".dump"

        self.graph_creator = Graphing(config, "Simulation " + config.product)

        self.all_sims = DataLoader().load_sim_data(self.sim_root, 0, config.num_simulators)

    def show_graphs(self):
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
            self.config.plt.show()
            i += 1

    @classmethod
    def dump_confidence_data(cls, dst, time_prices_dict: dict, time_confidence_dict: dict):
        cls.logger.info("Dumping confidence data to " + dst)
        with open(dst, 'w', newline='\n') as fd:
            fd.write(str(time_prices_dict))
            fd.write(str(time_confidence_dict))

        cls.logger.info("Confidence data dumped to: " + dst)

    # TODO: fix some of these awful names, such as "seconds"
    def calculate_trade_percentiles(self, seconds_list: List[int], level=0.95):
        time_prices_dict = self.extract_trade_prices_at_times(self.all_sims, seconds_list)
        time_percentiles_dict = self.calculate_percentiles(time_prices_dict, level)

        self.dump_confidence_data(self.confidence_path, time_prices_dict, time_percentiles_dict)

        return time_percentiles_dict

    # TODO: reduce duplication
    def calculate_midprice_percentiles(self, time_prices_dict, level=0.95):
        time_percentiles_dict = self.calculate_percentiles(time_prices_dict, level)

        self.dump_confidence_data(self.confidence_path, time_prices_dict, time_percentiles_dict)

        return time_percentiles_dict

    @staticmethod
    def extract_prices_at_times(prices_dfs: List[pd.DataFrame], seconds_list):
        time_prices_dict = {}
        for seconds in seconds_list:
            time_prices_dict[seconds] = []
        sim_index = 0
        for price_df in prices_dfs:
            for seconds in seconds_list:
                price = DataUtils.get_last_price_before(price_df, seconds)
                time_prices_dict[seconds].append(price)
            sim_index += 1
        return time_prices_dict

    @staticmethod
    def extract_trade_prices_at_times(all_sims, seconds_list):
        prices_dfs = []
        for sim in all_sims:
            trades_dd = sim[1]
            trades_df = trades_dd.compute()
            if len(trades_df) == 0:
                continue
            trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
            prices_dfs.append(trades_df)

        return SimulationAnalysis.extract_prices_at_times(prices_dfs, seconds_list)

    # TODO: refactor this to use extract_prices_at_times function
    @staticmethod
    def extract_mid_prices_at_times(all_sims, seconds_list):
        time_prices_dict = {}
        for seconds in seconds_list:
            time_prices_dict[seconds] = []
        sim_index = 0
        for sim in all_sims:
            midprices_dd = sim[3]
            midprices_df = midprices_dd.compute()
            midprices_df['time'] = DataUtils().get_times_in_seconds_after_start(midprices_df['time'])
            for seconds in seconds_list:
                price = DataUtils.get_last_price_before(midprices_df, seconds)
                time_prices_dict[seconds].append(price)
            sim_index += 1
        return time_prices_dict

    @staticmethod
    def get_percentiles(data, level=0.025):
        n = len(data)
        data.sort()

        lower = data[int(n * level)]
        median = data[int(n / 2)]
        upper = data[int(n * (1 - level))]

        return lower, median, upper

    @classmethod
    def calculate_percentiles(cls, time_prices_dict, level: float):
        time_percentiles_dict = {}
        num_nans = {}
        low_high = {}
        for time_in_seconds, prices in time_prices_dict.items():
            filt_prices = list(filter(lambda p: not math.isnan(p), prices))
            low_high[time_in_seconds] = (min(filt_prices), max(filt_prices))
            num_nans[time_in_seconds] = len(prices) - len(filt_prices)
            interval = SimulationAnalysis.get_percentiles(filt_prices)
            time_percentiles_dict[time_in_seconds] = interval

        for key in num_nans:
            cls.logger.info(str(key) + " has " + str(num_nans[key]) + " NaNs")

        for key, lh in low_high.items():
            cls.logger.info(str(key) + " low, high: " + str(lh))

        for key, intervals in time_percentiles_dict.items():
            cls.logger.info(str(key) + " has percentile width: " + str(abs(intervals[0] - intervals[2])))

        return time_percentiles_dict
