import datetime
import dask.dataframe as dd
import stats
import graphs
import matplotlib.pyplot as plt


def reconstruct_orderbook(feed: dd, at: datetime.datetime) -> dd:
    """Reconstructs the state of the order book at a given time"""
    # Filter out orders/trades which arrived after the time we are interested in
    valid_messages = feed[feed['time'] < at.isoformat()]

    trades = stats.get_trades(valid_messages)
    cancellations = stats.get_cancellations(valid_messages)
    orders = stats.get_orders(valid_messages)

    # Find those orders which are no longer on the book
    # TODO: find those orders which were modified, handle carefully
    executed_order_ids = trades['order_id'].unique().compute()
    cancelled_order_ids = cancellations['order_id'].unique().compute()
    print(executed_order_ids)

    # Find those orders which are still on the book
    remaining_orders = orders[~orders['order_id'].isin(executed_order_ids)
                            & ~orders['order_id'].isin(cancelled_order_ids)].compute()

    return remaining_orders.reset_index(drop=True)

file = "/Users/jamesprince/project-data/merge/00:02:48.142841.parquet"

feed = dd.read_parquet(file)
time = datetime.datetime.utcnow()
orderbook = reconstruct_orderbook(feed, time)
print(orderbook)

graphs.graph_sides(orderbook, "BTC-USD")
plt.show()
