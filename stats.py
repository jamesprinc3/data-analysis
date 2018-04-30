import dask.dataframe as dd
import pandas as pd


class Statistics:

    def __init__(self):
        """
        A class which spits out a bunch of statistics related to trade feed data
        """

    @staticmethod
    def get_side(side: str, df: dd) -> dd:
        return df[df['side'] == side]

    @staticmethod
    def get_trades(df: dd) -> dd:
        return df[df['reason'] == 'filled']

    @staticmethod
    def get_orders(df: dd) -> dd:
        return df[df['type'] == 'received']

    @staticmethod
    def get_cancellations(df: dd) -> dd:
        return df[df['reason'] == 'canceled']

    def modifications(self, df: dd):
        pass

    @staticmethod
    def get_num_reason(reason: str, df: dd):
        num = len(df[df['reason'] == reason])
        return num

    @staticmethod
    def get_num_type(t: str, df: dd) -> int:
        num = len(df[df['type'] == t])
        return num

    @staticmethod
    def get_mean(col_name: str, df: dd) -> dd:
        return df[col_name].astype('float64').mean()

    @staticmethod
    def get_std_dev(col_name: str, df: dd) -> dd:
        return df[col_name].astype('float64').std()

    @staticmethod
    def get_buy_sell_ratio(df: dd) -> (float, float):
        num_buys = len(df[df['side'] == 'sell'])
        num_sells = len(df[df['side'] == 'buy'])

        buy_ratio = (100 * num_buys) / (num_buys + num_sells)
        sell_ratio = (100 * num_sells) / (num_buys + num_sells)

        return buy_ratio, sell_ratio

    # PRE: df must be trades only
    @staticmethod
    def get_price_over_time(trades_df: dd) -> pd.DataFrame:
        price_times_df = trades_df[['time', 'price']].dropna()
        price_times_df.rename(index=str, columns={"price": "most_recent_trade_price"}, inplace=True)
        print(price_times_df)
        price_times_df['most_recent_trade_price'] = price_times_df['most_recent_trade_price'].astype('float64')
        return price_times_df.drop_duplicates()

    def calculate_feed_stats(self, df: dd) -> None:
        """Calculate and print some statistics based on the data"""
        num_total_msgs = get_total(df)
        num_trades = self.get_num_reason('filled', df)
        num_cancel = self.get_num_reason('canceled', df)

        num_received = self.get_num_type('received', df)
        num_open = self.get_num_type('open', df)
        num_done = self.get_num_type('done', df)
        num_match = self.get_num_type('match', df)
        num_change = self.get_num_type('change', df)

        avg_trade_price = self.get_mean('price', self.get_trades(df))
        std_dev_trade_price = self.get_std_dev('price', self.get_trades(df))

        print("percentage of received messages: " + str((100 * num_received) / num_total_msgs) + "%")
        print("percentage of open messages: " + str((100 * num_open) / num_total_msgs) + "%")
        print("percentage of done messages: " + str((100 * num_done) / num_total_msgs) + "%")
        print("percentage of match messages: " + str((100 * num_match) / num_total_msgs) + "%")
        print("percentage of change messages: " + str((100 * num_change) / num_total_msgs) + "%")

        print("percentage of orders canceled: " + str((100 * num_cancel) / num_received) + "%")
        print("percentage of orders filled: " + str((100 * num_trades) / num_received) + "%")

        print("average trade price: " + str(avg_trade_price))
        print("std. dev. of trade price: " + str(std_dev_trade_price))

        self.calculate_stats(df)

    def calculate_stats(self, df:dd) -> None:

        buy_ratio, sell_ratio = self.get_buy_sell_ratio(df)
        print("Ratio (buy/sell): " + str(buy_ratio) + ":" + str(sell_ratio))

        avg_order_size = self.get_mean('size', df)
        std_dev_order_size = self.get_std_dev('size', df)

        avg_sell_order_size = self.get_mean('size', self.get_side('sell', df))
        std_dev_sell_order_size = self.get_std_dev('size', self.get_side('sell', df))

        avg_buy_order_size = self.get_mean('size', self.get_side('buy', df))
        std_dev_buy_order_size = self.get_std_dev('size', self.get_side('buy', df))

        avg_price = df['price'].astype('float64').mean()
        std_dev_price = df['price'].astype('float64').std()

        avg_sell_order_price = self.get_mean('price', self.get_side('sell', df))
        std_dev_sell_price = self.get_std_dev('price', self.get_side('sell', df))

        avg_buy_price = self.get_mean('price', self.get_side('buy', df))
        std_dev_buy_order_price = self.get_std_dev('price', self.get_side('buy', df))

        print("average order size: " + str(avg_order_size))
        print("std. dev. of order size: " + str(std_dev_order_size))

        print("average sell order size: " + str(avg_sell_order_size))
        print("sell order std. dev: " + str(std_dev_sell_order_size))

        print("average buy order size: " + str(avg_buy_order_size))
        print("buy order std. dev: " + str(std_dev_buy_order_size))

        print("average price: " + str(avg_price))
        print("std. dev. of price: " + str(std_dev_price))

        print("average sell order price: " + str(avg_sell_order_price))
        print("std. dev. of sell order price: " + str(std_dev_sell_price))

        print("average buy order price: " + str(avg_buy_price))
        print("std. dev. of buy order price: " + str(std_dev_buy_order_price))




def get_total(df: dd) -> int:
    total = len(df.values)
    print("number of rows: " + str(total))
    return len(df.values)
