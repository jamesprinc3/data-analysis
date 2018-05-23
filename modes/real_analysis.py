from typing import List

import dask.dataframe as dd
import matplotlib.pyplot as plt

from data.data_utils import DataUtils
from output.graphing import Graphing


class RealAnalysis:
    def __init__(self, data_description: str):

        self.data_description = data_description

    def generate_graphs(self, orders_df: dd, trades_df: dd, cancels_df: dd):
        graph_creator = Graphing("Real BTC-USD")

        graph_creator.graph_sides(orders_df)
        graph_creator.graph_relative_price_distribution(trades_df, orders_df, 100)
        graph_creator.graph_interval(orders_df)
        graph_creator.graph_price_time(trades_df, "Price over time")

        plt.show()

    def get_prices_at_times(self, seconds_list: List[int]):
        map(lambda x: DataUtils.get_last_price_before(self.trades_df, x), seconds_list)
