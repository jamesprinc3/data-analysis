import numpy as np
import dask.dataframe as dd

from data.data_splitter import DataSplitter
from data.data_transformer import DataTransformer


class Correlations:

    @staticmethod
    def get_price_size_corr(trades_df: dd, limit_orders: dd):
        ret = {}

        for side in ["buy", "sell"]:
            side_df = DataSplitter.get_side(side, limit_orders)

            prices = DataTransformer.get_relative_prices(trades_df, side_df)
            sizes = side_df[side_df['size'].index.isin(prices.index)]['size']

            if side == "buy":
                prices = prices.apply(lambda x: -x)

            ret[side] = Correlations.get_correlation_matrix(prices, sizes)[0, 1]

        return ret

    @staticmethod
    def get_correlation_matrix(x, y):
        return np.corrcoef(x, y)
