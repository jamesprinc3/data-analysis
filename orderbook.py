import datetime
import json
import logging
import pathlib
from os import listdir
from os.path import isfile, join
from typing import List

import dask.dataframe as dd
import pandas as pd

from data.data_splitter import DataSplitter


class OrderBook:
    logger = logging.getLogger()

    @staticmethod
    def get_spread(ob_state: dd) -> float:
        best_bid = DataSplitter.get_side("buy", ob_state).sort_values(by='price', ascending=False)['price'].max()
        best_ask = DataSplitter.get_side("sell", ob_state).sort_values(by='price')['price'].min()

        return best_ask - best_bid

    # TODO: move to DataLoader class?
    @staticmethod
    def enum_all_files(path: str) -> List[str]:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        onlyfiles.sort()
        return onlyfiles

    @staticmethod
    def locate_closest_ob_state(root: str, time: datetime.datetime):
        files = OrderBook.enum_all_files(root)
        files_before_time = list(filter(lambda f: f < time.isoformat(), files))
        closest_state = files_before_time[-1]

        return root + closest_state

    @staticmethod
    def load_orderbook_state(path: str):
        with open(path) as f:
            data = json.load(f)
            seq_num = data['sequence']
            bids = map(lambda row: (seq_num, "buy", row[0], row[1], row[2]), data['bids'])
            asks = map(lambda row: (seq_num, "sell", row[0], row[1], row[2]), data['asks'])

            all_active_orders = list(bids) + list(asks)
            column_names = ['sequence', 'side', 'price', 'size', 'order_id']
            ob_state = pd.DataFrame(all_active_orders, columns=column_names)
            ob_state[['price', 'size']] = ob_state[['price', 'size']].apply(pd.to_numeric)

            return ob_state

    @staticmethod
    def get_orderbook(orders: dd, trades: dd, cancels: dd, ob_state: dd) -> dd:
        """Gets those orders which are still active at the end of the feed"""
        # Find those orders which are no longer on the book
        # TODO: find those orders which were modified, handle carefully
        seq_num = ob_state['sequence'].iloc[0]
        limit_orders = DataSplitter.get_limit_orders(orders)
        residual_orders = limit_orders[limit_orders['sequence'] > seq_num]
        all_orders = ob_state.append(residual_orders)[['side', 'order_id', 'price', 'size']]

        executed_order_ids = trades['order_id'].unique()
        cancelled_order_ids = cancels['order_id'].unique()

        # Find those orders which are still on the book
        ob_filtered = all_orders[~all_orders['order_id'].isin(executed_order_ids)
                                 & ~all_orders['order_id'].isin(cancelled_order_ids)]

        # This variable is used in the pandas query below
        final_trade_price = trades['price'].dropna().iloc[-1]

        ob_final = DataSplitter.get_side("buy", ob_filtered).query('price < @final_trade_price').append(
            DataSplitter.get_side("sell", ob_filtered).query('price > @final_trade_price')
        )

        # if not OrderBook.check_ob_valid(ob_final):
        #     raise AssertionError("OrderBook does not appear to be valid")

        return ob_final.reset_index(drop=True)

    @staticmethod
    def check_ob_valid(ob: dd) -> bool:
        highest_buy = DataSplitter.get_side("buy", ob)['price'].max()
        lowest_sell = DataSplitter.get_side("sell", ob)['price'].min()

        return highest_buy < lowest_sell

    @staticmethod
    def orderbook_to_file(orderbook: pd.DataFrame, file_path: str):
        orderbook.sort_values(by='price', inplace=True)
        orderbook.to_csv(file_path)
