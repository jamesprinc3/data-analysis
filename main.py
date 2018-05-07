import argparse
import datetime
import logging
from logging.config import fileConfig

import dask.dataframe as dd

from analysis.real_analysis import RealAnalysis
from analysis.simulation_analysis import SimulationAnalysis
from analysis.combined_analysis import CombinedAnalysis
from data_loader import DataLoader


# https://stackoverflow.com/a/36194213
def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def add_boolean_argument(parser, name, default=False, help=""):
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')


if __name__ == "__main__":
    fileConfig('logging_config.ini')
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Analyse level III order book data')
    parser.add_argument('--real_root', metavar='-rd', type=str, nargs='?',
                        help='file path to real data')
    parser.add_argument('--sim_root', metavar='-sd', type=str, nargs='?',
                        help='file path to the root of simulation data')
    # parser.add_argument('--combined', metavar='y/n', type=str, nargs='?',
    #                     help='(default no) boolean of whether to include combined analysis')
    parser.add_argument('--start_time', metavar='YYYY-MM-DDTHH:mm:SS', type=str, nargs='?',
                        help='time to start the simulation from')
    parser.add_argument('--sampling_window', metavar='int', type=int, nargs='?',
                        help='number of seconds before start_time to sample from')
    parser.add_argument('--simulation_window', metavar='int', type=int, nargs='?',
                        help='number of seconds after start_time to simulate')
    parser.add_argument('--product', metavar='BTC-USD', type=str, nargs='?',
                        help='name of the product being traded (e.g. "BTC-USD")')
    parser.add_argument('--dist_path', metavar='Path', type=str, nargs='?',
                        help='path to where you wish to output distribution params')
    add_boolean_argument(parser, "combined", False, "(default no) boolean of whether to include combined analysis")
    add_boolean_argument(parser, "graphs", False, "whether to display graphs")
    add_boolean_argument(parser, "fit_distributions", False, "whether to fit distributions")
    add_boolean_argument(parser, "run_simulation", False, "whether to run the simulation (by spawning the "
                                                          "OrderBookSimulator")
    parser.print_help()

    args = parser.parse_args()
    real_root = args.real_root
    sim_root = args.sim_root
    start_time = args.start_time
    sampling_window = args.sampling_window
    simulation_window = args.simulation_window
    combined = args.combined
    product = args.product
    dist_path = args.dist_path
    graphs = args.graphs
    fit_distributions = args.fit_distributions
    run_simulation = args.run_simulation

    print(combined)
    print(args)

    if combined and sim_root and real_root and start_time and sampling_window and simulation_window and product:
        start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")

        combined_analysis = CombinedAnalysis(sim_root, real_root, start_time, sampling_window, simulation_window,
                                             product)

        if run_simulation:
            combined_analysis.run_simulation()
        if graphs:
            combined_analysis.graph_real_prices_with_simulated_confidence_intervals()
    else:
        if real_root and start_time and sampling_window and product:
            start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%dT%H:%M:%S")
            sampling_window_start_time = start_time - datetime.timedelta(seconds=sampling_window)
            sampling_window_end_time = start_time
            orders_df, trades_df, cancels_df = DataLoader.load_sampling_data(real_root, sampling_window_start_time,
                                                                             sampling_window_end_time, product)
            real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined BTC-USD")

            if fit_distributions and dist_path:
                params = real_analysis.generate_order_params()
                real_analysis.params_to_file(params, dist_path)
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
