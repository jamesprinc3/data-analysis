import argparse
import logging
from logging.config import fileConfig

import dask.dataframe as dd
from real_analysis import RealAnalysis
from simulation_analysis import SimulationAnalysis

if __name__ == "__main__":
    fileConfig('logging_config.ini')
    logger = logging.getLogger()
    logger.debug('often makes a very good meal of %s', 'visiting tourists')

    parser = argparse.ArgumentParser(description='Analyse level III order book data')
    parser.add_argument('--real_data', metavar='-rd', type=str, nargs='?',
                        help='file path to real data for which you want some info/statistics')
    parser.add_argument('--sim_data', metavar='-sd', type=str, nargs='?',
                        help='file path to simulation data for which you want some info/statistics')
    parser.print_help()

    args = parser.parse_args()
    real_data_file_path = args.real_data
    sim_data_file_path = args.sim_data

    if real_data_file_path:
        RealAnalysis(real_data_file_path, "BTC-USD").task()

    if sim_data_file_path:
        SimulationAnalysis(sim_data_file_path, "BTC-USD").analyse()


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    print("number of sell side interactions: " + str(num_sells))
    print("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells



