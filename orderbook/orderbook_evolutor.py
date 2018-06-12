import datetime
import heapq
from typing import List

import pandas as pd

from data.data_splitter import DataSplitter


class OrderBookEvolutor:

    def __init__(self, start_ob_state_df: pd.DataFrame, st: datetime.datetime, start_seq: int):
        self.column_order = ['price', 'order_id', 'side', 'size']

        bids = DataSplitter.get_side("buy", start_ob_state_df)
        # have to change the sign of the prices column so that we can use a min heap as a max heap...
        bids['price'] = bids['price'].apply(lambda x: -x)
        bids = bids[self.column_order]
        bids = bids.values.tolist()
        self.bids_max_heap = list(map(tuple, bids))
        self.bids_max_heap.sort()
        heapq.heapify(self.bids_max_heap)

        asks = DataSplitter.get_side("sell", start_ob_state_df)
        asks = asks[self.column_order]
        asks = asks.values.tolist()
        self.asks_min_heap = list(map(tuple, asks))
        self.asks_min_heap.sort()
        heapq.heapify(self.asks_min_heap)

        # Keep track of order ids which should no longer be on the book
        self.invalid_order_ids = set()

        self.st = st

        self.start_seq = start_seq

    def evolve_orderbook_discrete(self, feed_df: pd.DataFrame, step_seconds: int, max_events=None):
        """"Move the state of the order book forwards in time and output spread and midprice for every step"""
        et = feed_df['time'].iloc[-1]
        num_samples = int((et - self.st).total_seconds() / step_seconds)

        sample_index = 0

        sample_times = []
        best_bids = []
        best_asks = []
        midprices = []
        spreads = []

        feed_df = feed_df[feed_df['sequence'] > self.start_seq]

        indices = range(0, len(feed_df))
        if max_events:
            indices = range(0, max_events)

        first_sec_df = feed_df[feed_df['time'] < self.st + datetime.timedelta(seconds=2)]
        first_sec_data = list(map(tuple, first_sec_df.values.tolist()))

        for event in first_sec_data:
            print(event)

        for i in indices:
            event = feed_df.iloc[i]

            sample_time = self.st + datetime.timedelta(seconds=sample_index * step_seconds)
            if sample_time < event['time']:
                best_bids.append(self.get_best_price("buy"))
                best_asks.append(self.get_best_price("sell"))
                midprices.append(self.get_midprice())
                spreads.append(self.get_spread())
                sample_times.append(sample_time)
                sample_index += 1

            # We can ignore market orders since their effect comes from the trade data
            if OrderBookEvolutor.is_open(event):
                self.add_order(event)
            elif OrderBookEvolutor.is_done(event):
                self.remove_order(event)
            # elif OrderBookEvolutor.is_cancel(event):
            #     self.remove_order(event)

        return pd.DataFrame({'time': sample_times, 'best_bid': best_bids, 'best_ask': best_asks,
                             'midprice': midprices, 'spread': spreads})

    def evolve_orderbook(self, feed_df: pd.DataFrame, max_events=None):
        """"Move the state of the order book forwards in time and output spread and midprice for every step"""

        event_times = []
        event_seqs = []
        best_bids = []
        best_asks = []
        midprices = []
        spreads = []

        feed_df = feed_df[feed_df['sequence'] > self.start_seq]
        feed_df = feed_df[feed_df['type'].isin(['open', 'done'])]
        feed_df['size'] = feed_df['remaining_size']

        indices = range(0, len(feed_df))
        if max_events:
            indices = range(0, max_events)

        for i in indices:
            event = feed_df.iloc[i]

            # Record the metrics
            best_bids.append(self.get_best_price("buy"))
            best_asks.append(self.get_best_price("sell"))
            midprices.append(self.get_midprice())
            spreads.append(self.get_spread())
            event_times.append(event['time'])
            event_seqs.append(event['sequence'])

            # We can ignore market orders since their effect comes from the trade data
            if OrderBookEvolutor.is_open(event):
                self.add_order(event)
            elif OrderBookEvolutor.is_done(event):
                self.remove_order(event)

        return pd.DataFrame({'sequence': event_seqs, 'time': event_times, 'best_bid': best_bids,
                             'best_ask': best_asks, 'midprice': midprices, 'spread': spreads})

    def get_spread(self):
        return self.get_best_price("sell") - self.get_best_price("buy")

    def get_midprice(self):
        return (self.get_best_price("sell") + self.get_best_price("buy")) / 2

    def get_best_price(self, side: str):
        if side == "buy":
            self.bids_max_heap = self.remove_invalid_orders(self.bids_max_heap)
            return -self.bids_max_heap[0][0]
        if side == "sell":
            self.asks_min_heap = self.remove_invalid_orders(self.asks_min_heap)
            return self.asks_min_heap[0][0]

    def remove_invalid_orders(self, heap):
        head_order = heap[0]
        head_order_id = head_order[1]

        while head_order_id in self.invalid_order_ids:
            heapq.heappop(heap)
            self.invalid_order_ids.remove(head_order_id)

        return heap

    def add_order(self, limit_order):
        limit_order = OrderBookEvolutor.reduce_columns(limit_order, self.column_order)
        if limit_order['side'] == 'buy':
            # Flip for max heap
            limit_order['price'] = -limit_order['price']
            heapq.heappush(self.bids_max_heap, tuple(limit_order))
        elif limit_order['side'] == 'sell':
            heapq.heappush(self.asks_min_heap, tuple(limit_order))

    def remove_order(self, event):
        self.invalid_order_ids.add(event['order_id'])
        self.invalid_order_ids.add(event['maker_order_id'])

    @staticmethod
    def reduce_columns(event, column_order: List[str]):
        return event[column_order]

    @staticmethod
    def is_open(event):
        return event['type'] == 'open'

    @staticmethod
    def is_done(event):
        return event['type'] == 'done'
    #
    # @staticmethod
    # def is_trade(event):
    #     return event['reason'] == 'filled'
    #
    # @staticmethod
    # def is_cancel(event):
    #     return event['reason'] == 'canceled'
