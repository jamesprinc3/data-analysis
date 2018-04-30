import pandas as pd
from graph_creator import GraphCreator
import matplotlib.pyplot as plt
import os
from data_splitter import DataSplitter
from stats import Statistics
from data_loader import DataLoader


class SimulationAnalysis:
    def __init__(self, root, data_desc):
        """
        :param root: The data root for the output from OrderBookSimulator
        :param data_desc: A description of what data is being analysed (to be eventually outputted on graphs)
        """
        self.root = root
        self.graph_creator = GraphCreator("Simulation " + data_desc)

    def analyse(self):
        # print(self.dirs)
        # for directory in self.dirs:
        orders_dd, trades_dd, cancels_dd = DataLoader().load_sim_data(self.root)

        orders_df = orders_dd.compute()
        # self.graph_creator.graph_order_sizes(orders_df)
        # self.graph_creator.graph_price_quantity(orders_df)
        self.graph_creator.graph_price_time(orders_df, "orders")
        # self.graph_creator.graph_time_delta(orders_df)
        Statistics().calculate_stats(orders_df)
        #
        trades_df = trades_dd.compute()
        self.graph_creator.graph_price_time(trades_df, "trades")

        # cancels_df = pd.read_csv(cancels_path)
        # print("cancels df")
        # print(cancels_df)
        # self.graph_creator.graph_relative_price_distribution(trades_df, cancels_df, 20)

        self.graph_creator.graph_relative_price_distribution(trades_df, orders_df, 20)

        # graphs.graph_price_quantity(trades_df)

        plt.show()

