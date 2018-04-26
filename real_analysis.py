import dask.dataframe as dd
from stats import Statistics
from graph_creator import GraphCreator
import pandas as pd
import matplotlib.pyplot as plt


class RealAnalysis():

    def __init__(self, file_path, data_description):
        self.file_path = file_path
        self.data_description = data_description

    def task(self):
        input_df = dd.read_parquet(self.file_path).compute()
        btc_usd_df = input_df[input_df['product_id'] == 'BTC-USD']

        btc_usd_price_buy = pd.Series(Statistics.get_side('buy', input_df)['price'].astype('float64').tolist())

        sample_size = 100
        std_devs = 3

        data = Statistics.keep_n_std_dev(btc_usd_price_buy, std_devs)
        if len(btc_usd_price_buy) > sample_size:
            data = btc_usd_price_buy.sample(n=sample_size)

        # TODO: replace this with real data_description
        GraphCreator("BTC-USD trades").graph_price(Statistics.get_trades(btc_usd_df))

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


