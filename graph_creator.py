import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd

from distribution_fitter import DistributionFitter
from stats import Statistics


class GraphCreator:

    def __init__(self, data_desc: str):
        self.data_description = data_desc

    def graph_time_delta(self, orders_df: dd):
        order_time_delta_df = orders_df['time'].apply(lambda x: self.date_to_unix(x, 'ns') / 1e6).diff()
        print(order_time_delta_df)
        cleaned_df = order_time_delta_df[order_time_delta_df != 0]
        self.get_distribution(cleaned_df, 'BTC-USD inter-order arrival times', 'inter order time (ms)', bins=100)

    def graph_order_sizes(self, orders_df: dd):
        size_df = pd.Series(orders_df['size'].astype('float64').tolist())
        self.get_distribution(size_df, 'BTC-USD Order size', 'Order Size', 2000)

    def graph_trade_sizes(self, trades_df: dd):
        order_id_map = trades_df[['order_id', 'size']].drop_duplicates()
        traded_order_id_set = pd.DataFrame(trades_df['order_id'].unique(), columns=['order_id'])
        joined = order_id_map.join(traded_order_id_set.set_index('order_id'), on='order_id', how='inner')
        result = pd.Series(joined.dropna(axis=0, how='any').reset_index(drop=True)['size'].astype('float64').tolist())
        self.get_distribution(result, 'BTC-USD Trade Order Size', 'Trade Order Size')

    def graph_sides(self, df: dd, product: str) -> None:
        btc_usd_price_buy = pd.Series(Statistics().get_side('buy', df)['price'].astype('float64').tolist())
        btc_usd_price_sell = pd.Series(Statistics().get_side('sell', df)['price'].astype('float64').tolist())

        self.get_distribution(btc_usd_price_buy, product + ' buy side', 'Price ($)', bins=50)
        self.get_distribution(btc_usd_price_sell, product + ' sell side', 'Price ($)', bins=50)

    def graph_price(self, df: dd):
        #
        y = df['price'].astype('float64').fillna(method='ffill')
        print(y)
        x = df['time'].astype('datetime64[ns]').apply(lambda x: self.date_to_unix(x, 'ns') / 1e6)
        print(x)

        plt.figure(figsize=(12, 8))

        plt.plot(x, y, 'r+')

        plt.xlabel('Time (ns since epoch)')
        plt.ylabel('Price ($)')
        plt.ylim(8000, 10000)

        plt.title(self.data_description + ' price')

    # PRE: assume that the incoming df is either all trades or all orders (not sure the data will make much sense
    # otherwise
    @staticmethod
    def graph_price_quantity(df: dd) -> None:
        prices = df['price'].astype('float64').tolist()
        quantities = df['size'].astype('float64').tolist()

        plt.figure(figsize=(12, 8))
        plt.scatter(prices, quantities, marker='+')

    @staticmethod
    def date_to_unix(s, unit: str):
        return pd.to_datetime(s, unit=unit).value

    @staticmethod
    def get_distribution(data: pd.Series, description: str, xlabel: str, bins=20, std_devs: int=2):
        sample_size = 10000

        data = Statistics().keep_n_std_dev(data, std_devs)
        if len(data) > sample_size:
            data = data.sample(n=sample_size)
        data = Statistics().keep_n_std_dev(data, std_devs)

        DistributionFitter().best_fit_with_graphs(data, description, xlabel, bins=bins)