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
        for directory in self.dirs[0]:
            orders_path: str = self.data_root + directory + "/orders.csv"
            trades_path: str = self.data_root + directory + "/trades.csv"

            # orders_df = pd.read_csv(order_path)
            # graphs.graph_order_sizes(orders_df)

            trades_df = pd.read_csv(trades_path)
            self.graph_creator.graph_price(trades_df)

            orders_df = pd.read_csv(orders_path)
            self.graph_creator.graph_price(orders_df)

            # graphs.graph_price_quantity(trades_df)

        plt.show()
