import datetime
import dask.dataframe as dd
import stats
from graph_creator import GraphCreator
import matplotlib.pyplot as plt
import pandas as pd
from stats import Statistics

class OrderBook:

    @staticmethod
    def reconstruct_orderbook(feed: dd, at: datetime.datetime) -> dd:
        """Reconstructs the state of the order book at a given time"""
        # Filter out orders/trades which arrived after the time we are interested in
        valid_messages = feed[feed['time'] < at.isoformat()]

        trades = Statistics.get_trades(valid_messages)
        print("trade min: " + str(trades['price'].astype('float').min()))
        print("trade max: " + str(trades['price'].astype('float').max()))
        cancellations = Statistics.get_cancellations(valid_messages)
        orders = Statistics.get_orders(valid_messages)

        # Find those orders which are no longer on the book
        # TODO: find those orders which were modified, handle carefully
        executed_order_ids = trades['order_id'].unique()
        cancelled_order_ids = cancellations['order_id'].unique()
        # print(executed_order_ids)

        # Find those orders which are still on the book
        remaining_orders = orders[~orders['order_id'].isin(executed_order_ids)
                                  & ~orders['order_id'].isin(cancelled_order_ids)]

        return remaining_orders.reset_index(drop=True)

    @staticmethod
    def orderbook_to_file(orderbook: pd.DataFrame, file_path: str):
        orderbook.sort_values(by='price')
        orderbook.to_csv(file_path)

input_file = "/Users/jamesprince/project-data/2018-03-25.parquet"
output_file = "/Users/jamesprince/project-data/orderbook-2018-03-25-01:00:00.csv"

feed = dd.read_parquet(input_file)
btc_usd_feed = feed[feed['product_id'] == 'BTC-USD'].reset_index(drop=True).compute()
# print(btc_usd_feed)
time = datetime.datetime(year=2018, month=3, day=25, hour=1)
print(time)
orderbook = OrderBook.reconstruct_orderbook(btc_usd_feed, time)
# print(orderbook)
# stats.calculate_stats(orderbook)

graph_creator = GraphCreator("BTC-USD Order Book")

graph_creator.graph_sides(orderbook)
graph_creator.graph_order_sizes(orderbook)
graph_creator.graph_price_quantity(orderbook)

OrderBook.orderbook_to_file(orderbook, output_file)

plt.show()
