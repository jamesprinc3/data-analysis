import argparse
import logging
from logging.config import fileConfig

import dask.dataframe as dd
from real_analysis import RealAnalysis

if __name__ == "__main__":
    fileConfig('logging_config.ini')
    logger = logging.getLogger()
    logger.debug('often makes a very good meal of %s', 'visiting tourists')

    parser = argparse.ArgumentParser(description='Consolidate multiple parquet files into just one.')
    parser.add_argument('input_file', metavar='-i', type=str, nargs=1,
                        help='input file for which you want some info/statistics')

    args = parser.parse_args()
    file = args.input_file

    RealAnalysis(file, "BTC-USD").task()


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    print("number of sell side interactions: " + str(num_sells))
    print("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells



