import argparse
import configparser
import datetime
import logging
from logging.config import fileConfig

import dask.dataframe as dd

from analysis.real_analysis import RealAnalysis
from analysis.simulation_analysis import SimulationAnalysis
from analysis.combined_analysis import CombinedAnalysis
from data_loader import DataLoader

if __name__ == "__main__":
    fileConfig('logging_config.ini')
    logger = logging.getLogger()

    config = configparser.ConfigParser()
    config.read('combined.ini')

    real_root = config['paths']['real_root']
    sim_root = config['paths']['sim_root']
    params_path = config['paths']['params_path']

    sampling_window = config['window']['sampling']
    simulation_window = config['window']['simulation']

    start_time = config['data']['start_time']
    product = config['data']['product']

    combined = config['bools']['combined']
    graphs = config['bools']['graphs']
    run_simulation = config['bools']['run_simulation']
    fit_distributions = config['bools']['fit_distributions']

    if combined and sim_root and real_root and start_time and sampling_window and simulation_window and product and params_path:
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")

        combined_analysis = CombinedAnalysis(sim_root, real_root, start_time, sampling_window, simulation_window,
                                             product, params_path)

        if run_simulation:
            combined_analysis.run_simulation()
        if graphs:
            combined_analysis.print_stat_comparison()
            combined_analysis.graph_real_prices_with_simulated_confidence_intervals()
    else:
        if real_root and start_time and sampling_window and product:
            start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            sampling_window_start_time = start_time - datetime.timedelta(seconds=sampling_window)
            sampling_window_end_time = start_time
            orders_df, trades_df, cancels_df = DataLoader.load_sampling_data(real_root, sampling_window_start_time,
                                                                             sampling_window_end_time, product)
            real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined BTC-USD")

            if fit_distributions and params_path:
                params = real_analysis.generate_order_params()
                real_analysis.params_to_file(params, params_path)
            if graphs:
                real_analysis.generate_graphs()

        if sim_root:
            SimulationAnalysis(sim_root, "BTC-USD").analyse()


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    logger.debug("number of sell side interactions: " + str(num_sells))
    logger.debug("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells
