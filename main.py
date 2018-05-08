import argparse
import configparser
import datetime
import logging

import dask.dataframe as dd

from logging.config import fileConfig
from analysis.combined_analysis import CombinedAnalysis
from analysis.real_analysis import RealAnalysis
from analysis.simulation_analysis import SimulationAnalysis
from data_loader import DataLoader


def combined_mode(start_time: datetime.datetime):
    combined_analysis = CombinedAnalysis(sim_root, real_root, start_time, sampling_window, simulation_window,
                                         product, params_path)

    if run_simulation:
        combined_analysis.run_simulation()
    if graphs:
        combined_analysis.print_stat_comparison()
        combined_analysis.graph_real_prices_with_simulated_confidence_intervals()


def real_mode(start_time: datetime.datetime):
    sampling_window_start_time = start_time - datetime.timedelta(seconds=sampling_window)
    sampling_window_end_time = start_time
    orders_df, trades_df, cancels_df = DataLoader.load_sampling_data(real_root, sampling_window_start_time,
                                                                     sampling_window_end_time, product)
    real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined " + product)

    if fit_distributions and params_path:
        params = real_analysis.generate_order_params()
        real_analysis.params_to_file(params, params_path)
    if graphs:
        real_analysis.generate_graphs()


def simulation_mode():
    print(sim_root)
    SimulationAnalysis(sim_root, product).analyse()


if __name__ == "__main__":
    fileConfig('config/logging_config.ini')
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Analyse level III order book data')
    parser.add_argument('--config', metavar='path', type=str, nargs='?',
                        help='path to config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    real_root = config['paths']['real_root']
    sim_root = config['paths']['sim_root']
    params_path = config['paths']['params_path']

    sampling_window = int(config['window']['sampling'])
    simulation_window = int(config['window']['simulation'])

    start_time = config['data']['start_time']
    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    product = config['data']['product']

    mode = config['behaviour']['mode']
    graphs = config['behaviour']['graphs']
    run_simulation = config['behaviour']['run_simulation']
    fit_distributions = config['behaviour']['fit_distributions']

    print(mode)

    if mode == "combined":
        combined_mode(start_time)
    elif mode == "real":
        real_mode(start_time)
    elif mode == "simulation":
        simulation_mode()


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    logger.debug("number of sell side interactions: " + str(num_sells))
    logger.debug("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells
