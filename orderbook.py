import logging

import dask.dataframe as dd
import pandas as pd


class OrderBook:

    logger = logging.getLogger()

    @staticmethod
    def orderbook_from_df(orders: dd, trades: dd, cancels: dd) -> dd:
        """Reconstructs the state of the order book at the end of the feed"""
        # Find those orders which are no longer on the book
        # TODO: find those orders which were modified, handle carefully
        executed_order_ids = trades['order_id'].unique()
        cancelled_order_ids = cancels['order_id'].unique()

        # Find those orders which are still on the book
        remaining_orders = orders[~orders['order_id'].isin(executed_order_ids)
                                  & ~orders['order_id'].isin(cancelled_order_ids)]

        return remaining_orders.reset_index(drop=True)

    @staticmethod
    def orderbook_to_file(orderbook: pd.DataFrame, file_path: str):
        orderbook.sort_values(by='price')
        orderbook.to_csv(file_path)




