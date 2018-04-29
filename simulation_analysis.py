import pandas as pd
from graph_creator import GraphCreator
import matplotlib.pyplot as plt
import os


class SimulationAnalysis:
    def __init__(self, root, data_desc):
        """

        :param root: The data root for the output from OrderBookSimulator
        :param data_desc: A description of what data is being analysed (to be eventually outputted on graphs)
        """
        self.data_root = root
        self.dirs = next(os.walk(self.data_root))[1]
        self.graph_creator = GraphCreator(data_desc)

    def analyse(self):
        print(self.dirs)
        for directory in self.dirs:
            orders_path: str = self.data_root + directory + "/orders.csv"
            trades_path: str = self.data_root + directory + "/trades.csv"
            cancels_path: str = self.data_root + directory + "/cancels.csv"

            orders_df = pd.read_csv(orders_path)
            # self.graph_creator.graph_order_sizes(orders_df)
            # self.graph_creator.graph_price_quantity(orders_df)
            # self.graph_creator.graph_price_time(orders_df)
            self.graph_creator.graph_time_delta(orders_df)
            #
            trades_df = pd.read_csv(trades_path)
            # self.graph_creator.graph_price_time(trades_df)

            # cancels_df = pd.read_csv(cancels_path)
            # print("cancels df")
            # print(cancels_df)
            # self.graph_creator.graph_relative_price_distribution(trades_df, cancels_df, 20)



            # graphs.graph_price_quantity(trades_df)

        plt.show()

