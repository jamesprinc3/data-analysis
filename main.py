import argparse
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

    parser = argparse.ArgumentParser(description='Analyse level III order book data')
    parser.add_argument('--real_root', metavar='-rd', type=str, nargs='?',
                        help='file path to real data')
    parser.add_argument('--sim_root', metavar='-sd', type=str, nargs='?',
                        help='file path to the root of simulation data')
    parser.add_argument('--combined', metavar='y/n', type=str, nargs='?',
                        help='(default no) boolean of whether to include combined analysis')
    parser.add_argument('--start_time', metavar='YYYY-MM-DDTHH:mm:SS', type=str, nargs='?',
                        help='time to start the simulation from')
    parser.add_argument('--sampling_window', metavar='int', type=int, nargs='?',
                        help='number of seconds before start_time to sample from')
    parser.add_argument('--simulation_window', metavar='int', type=int, nargs='?',
                        help='number of seconds after start_time to simulate')
    parser.print_help()

    args = parser.parse_args()
    real_root = args.real_root
    sim_root = args.sim_root
    start_time = args.start_time
    sampling_window = args.sampling_window
    simulation_window = args.simulation_window
    combined = args.combined

    if combined == "y" and sim_root and real_root and start_time and sampling_window and simulation_window:
        start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")

        combined_analysis = CombinedAnalysis(sim_root, real_root, start_time, sampling_window, simulation_window)
        combined_analysis.run_simulation()
        combined_analysis.graph_real_prices_with_simulated_confidence_intervals()
    else:
        if real_root and start_time and sampling_window:
            start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
            sampling_window_start_time = start_time - datetime.timedelta(seconds=sampling_window)
            sampling_window_end_time = start_time
            orders_df, trades_df, cancels_df = DataLoader.load_sampling_data(real_root, sampling_window_start_time, sampling_window_end_time)
            real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined BTC-USD")
            # real_analysis.generate_order_params()
            real_analysis.generate_graphs()

        if sim_root:
            SimulationAnalysis(sim_root, "BTC-USD").analyse()


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    print("number of sell side interactions: " + str(num_sells))
    print("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells
