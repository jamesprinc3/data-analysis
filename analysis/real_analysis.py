from typing import List

import dask.dataframe as dd
import matplotlib.pyplot as plt

from data_utils import DataUtils
from graph_creator import GraphCreator


class RealAnalysis:
    def __init__(self, orders_df: dd, trades_df: dd, cancels_df: dd, data_description: str):
        if orders_df.empty or trades_df.empty or cancels_df.empty:
            raise AssertionError("At least one DataFrame is empty")

        self.orders_df = orders_df
        self.trades_df = trades_df
        self.cancels_df = cancels_df

        self.data_description = data_description

    def generate_graphs(self, orders_df: dd, trades_df: dd, cancels_df: dd):
        graph_creator = GraphCreator("Real BTC-USD")

        graph_creator.graph_sides(orders_df)
        graph_creator.graph_relative_price_distribution(trades_df, orders_df, 100)
        graph_creator.graph_interval(orders_df)
        graph_creator.graph_price_time(trades_df, "Price over time")

        plt.show()

    def get_prices_at_times(self, seconds_list: List[int]):
        map(lambda x: DataUtils.get_last_price_before(self.trades_df, x), seconds_list)
