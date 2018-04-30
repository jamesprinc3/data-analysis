import dask.dataframe as dd
from stats import Statistics
from graph_creator import GraphCreator
import pandas as pd
import matplotlib.pyplot as plt
from data_splitter import DataSplitter
from data_loader import DataLoader


class RealAnalysis:

    def __init__(self, file_path, data_description):
        self.file_path = file_path
        self.data_description = data_description

    @staticmethod
    def secs_to_nanos(secs) -> int:
        return secs * 10**9

    def task(self):
        input_dd = DataLoader().load_real_data(self.file_path)
        btc_usd_dd = input_dd[input_dd['product_id'] == 'BTC-USD']

        # btc_usd_price_buy = pd.Series(Statistics.get_side('buy', input_dd)['price'].astype('float64').tolist())
        #
        # sample_size = 100
        # std_devs = 3

        # data = Statistics.keep_n_std_dev(btc_usd_price_buy, std_devs)
        # if len(btc_usd_price_buy) > sample_size:
        #     data = btc_usd_price_buy.sample(n=sample_size)

        # TODO: replace this with real data_description
        graph_creator = GraphCreator("Real BTC-USD")
        num_seconds = 10

        orders_df = Statistics.get_orders(btc_usd_dd).compute()
        orders_df = DataSplitter.get_first_n_nanos(orders_df, self.secs_to_nanos(num_seconds))
        graph_creator.graph_price_time(orders_df, "orders")
        # Statistics().calculate_feed_stats(orders_df)


        # graph_creator.graph_price_time(Statistics.get_trades(btc_usd_df))
        # graph_creator.graph_order_sizes(orders_df)
        # graph_creator.graph_price_quantity(orders_df)
        # graph_creator.graph_time_delta(orders_df)

        # graph_creator.graph_order_cancel_relative_price_distribution(btc_usd_df)

        trades_df = Statistics.get_trades(btc_usd_dd).compute()

        trades_df = DataSplitter.get_first_n_nanos(trades_df, self.secs_to_nanos(num_seconds))
        graph_creator.graph_price_time(trades_df, "trades")

        # graph_creator.graph_sides(Statistics.get_orders(btc_usd_df))
        # Statistics().get_price_over_time(btc_usd_df)
        # graph_creator.graph_order_relative_price_distribution(btc_usd_df)
        # graph_creator.graph_time_delta(Statistics.get_orders(btc_usd_df))

        graph_creator.graph_relative_price_distribution(trades_df, orders_df, 100)

        plt.show()


    # num_total = total(df)
    # num_btc_usd = total(btc_usd_df)
    # print("percentage of orders of this market vs whole feed: " + str((100*num_btc_usd) / num_total) + "%")
    #



    # print(df['product_id'].unique())
    # print(df)


    # print(btc_usd_df)

    # trades = stats.get_trades(btc_usd_df)[['order_id', 'price']].dropna()
    # order_sizes = stats.get_orders(btc_usd_df)[['order_id', 'size']]
    # joined = trades.join(order_sizes.set_index('order_id'), how='inner', on='order_id')
    # print(trades)
    # print(order_sizes)
    # graphs.graph_price_quantity(joined)
    # plt.show()
    # stats.calculate_stats(btc_usd_df)
    #
    # graph_sides(btc_usd_df, "BTC-USD")


    # btc_usd_price_buy = keep_n_std_dev(btc_usd_price_buy, std_devs)
    # print("here")
    #
    # theoretical, _ = fitting.best_fit_with_graphs(data, 200)
    #
    # print(theoretical)
    #
    # qqplot.plot(btc_usd_price_buy, theoretical)
    #


