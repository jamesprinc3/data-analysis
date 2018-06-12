import datetime
import json
import logging
import pathlib
from os import listdir
from os.path import isfile, join
from typing import List

import dask.dataframe as dd
import pandas as pd

from data.data_loader import DataLoader
from data.data_splitter import DataSplitter


class OrderBookCreator:
    logger = logging.getLogger("OrderBook")

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
        files = OrderBookCreator.enum_all_files(root)
        files_before_time = list(filter(lambda f: f < time.isoformat(), files))
        closest_state_str = files_before_time[-1]

        closest_state_time = datetime.datetime.strptime(closest_state_str.rsplit(".", 2)[0], "%Y-%m-%dT%H:%M:%S")

        return closest_state_time, closest_state_str

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

            return seq_num, ob_state

    @staticmethod
    def get_orderbook(feed_df: pd.DataFrame, ob_state: dd, ob_state_seq) -> dd:
        """Gets those orders which are still active at the end of the feed"""
        # Find those orders which are no longer on the book
        # TODO: find those orders which were modified, handle carefully
        open_messages = feed_df[feed_df['type'] == 'open']
        open_messages['size'] = open_messages['remaining_size']
        print(open_messages)
        residual_orders = open_messages[open_messages['sequence'] > ob_state_seq]
        all_orders = ob_state.append(residual_orders)

        done_messages = feed_df[feed_df['type'] == 'done']
        done_order_ids = list(done_messages['order_id'])



        # Find those orders which are still on the book
        ob_filtered = all_orders[~all_orders['order_id'].isin(done_order_ids)]

        # This variable is used in the pandas query below
        # final_trade_price = trades['price'].dropna().iloc[-1]

        # ob_final = DataSplitter.get_side("buy", ob_filtered).query('price < @final_trade_price').append(
        #     DataSplitter.get_side("sell", ob_filtered).query('price > @final_trade_price')
        # )

        if not OrderBookCreator.check_ob_valid(ob_filtered):
            raise AssertionError("OrderBook does not appear to be valid")

        final_seq = ob_filtered['sequence'].sort_values().iloc[-1]

        return ob_filtered.reset_index(drop=True)[['side', 'order_id', 'price', 'size']], final_seq

    @staticmethod
    def check_ob_valid(ob: dd) -> bool:
        highest_buy = DataSplitter.get_side("buy", ob)['price'].max()
        lowest_sell = DataSplitter.get_side("sell", ob)['price'].min()

        return highest_buy < lowest_sell

    @classmethod
    def orderbook_to_file(cls, orderbook: pd.DataFrame, file_path: str) -> None:
        try:
            orderbook.sort_values(by='price', inplace=True)
            orderbook.to_csv(file_path, index=False)
            cls.logger.info("Orderbook saved to: " + file_path)
        except Exception as e:
            cls.logger.error("Failed to save orderbook to " + file_path + " exception: " + str(e))
            raise e

    @staticmethod
    def plot_orderbook(ob_state, xwindow, log_y_scale=False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        bids = DataSplitter.get_side("buy", ob_state)
        asks = DataSplitter.get_side("sell", ob_state)

        OrderBookCreator.__plot_bid_side(bids, xwindow, percentile=0.9)
        OrderBookCreator.__plot_ask_side(asks, xwindow, percentile=0.9)

        plt.title("Order Book")
        plt.xlabel("Price")
        plt.ylabel("Cumulative size")
        if log_y_scale:
            plt.yscale('log')
        plt.legend()
        plt.show()

    @staticmethod
    def __plot_bid_side(bids: pd.DataFrame, xwindow, percentile=0.8):
        import matplotlib.pyplot as plt

        xmin = bids['price'].max() - xwindow / 2

        bids.sort_values(by='price', ascending=False)

        running_total = 0
        xs = [bids['price'].iloc[0]]
        ys = [0]
        keep = int(percentile * (len(bids) - 1))

        for index in range(0, keep):
            if bids['price'].iloc[index] < xmin:
                break

            running_total += bids['size'].iloc[index]
            xs = xs + [bids['price'].iloc[index], bids['price'].iloc[index + 1]]
            ys = ys + [running_total, running_total]

        plt.plot(xs, ys, 'g', label="Bid Side")

    @staticmethod
    def __plot_ask_side(asks, xwindow, percentile=0.8):
        import matplotlib.pyplot as plt
        asks.sort_values(by='price', ascending=True)

        xmax = asks['price'].min() + xwindow / 2

        running_total = 0
        xs = [asks['price'].iloc[0]]
        ys = [0]
        keep = int(percentile * (len(asks) - 1))

        for index in range(0, keep):
            if asks['price'].iloc[index] > xmax:
                break

            running_total += asks['size'].iloc[index]
            xs = xs + [asks['price'].iloc[index], asks['price'].iloc[index + 1]]
            ys = ys + [running_total, running_total]

        plt.plot(xs, ys, 'r', label="Ask Side")

        pass


def reconstruct_orderbook(config, sim_st, logger):
    try:
        closest_state_time_utc_1, closest_state_str = OrderBookCreator.locate_closest_ob_state(
            config.orderbook_input_root,
            sim_st + datetime.timedelta(hours=1))
        # - 10 seconds so that we definitely get all of the messages
        closest_state_time_utc_0 = closest_state_time_utc_1 - datetime.timedelta(hours=1, seconds=10)
        feed_df = DataLoader.load_feed(config.real_root, closest_state_time_utc_0, sim_st, config.product)
        closest_state_file_path = config.orderbook_input_root + closest_state_str
        logger.info("Closest order book path: " + closest_state_file_path)
        ob_state_seq, ob_state_df = OrderBookCreator().load_orderbook_state(closest_state_file_path)
        logger.info("Orderbook state sequence: " + str(ob_state_seq))
        logger.info("Feed first sequence: " + str(feed_df['sequence'].values.min()))
        ob_final, ob_final_seq = OrderBookCreator().get_orderbook(feed_df, ob_state_df, ob_state_seq)
        return ob_final_seq, ob_final
    except Exception as e:
        logger.error("Order Book Reconstruction failed: " + str(e))
