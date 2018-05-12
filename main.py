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
from orderbook import OrderBook


def combined_mode(st: datetime.datetime):
    combined_analysis = CombinedAnalysis(sim_root, real_root, graphs_root,
                                         st, sampling_window, simulation_window,
                                         orderbook_window, product, params_path,
                                         save_graphs, show_graphs)

    if run_simulation:
        combined_analysis.run_simulation()
    # if graphs:
    #     combined_analysis.print_stat_comparison()
    #     combined_analysis.graph_real_prices_with_simulated_confidence_intervals()


def multi_combined_mode(start_time: datetime.datetime):
    num_predictions = int(config['data']['num_predictions'])
    interval = int(config['data']['interval'])

    for i in range(0, num_predictions):
        st = start_time + datetime.timedelta(seconds=interval * i)
        try:
            combined_mode(st)
        except:
            print("we haz error")






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


def orderbook_mode():
    orderbook_window_start_time = start_time - datetime.timedelta(seconds=orderbook_window)
    orderbook_window_end_time = start_time

    orders_df, trades_df, cancels_df = DataLoader.load_sampling_data(real_root, orderbook_window_start_time,
                                                                     orderbook_window_end_time, product)

    orderbook = OrderBook.orderbook_from_df(orders_df, trades_df, cancels_df)
    output_file = "/Users/jamesprince/project-data/orderbook-" + orderbook_window_end_time.isoformat() + ".csv"
    OrderBook.orderbook_to_file(orderbook, output_file)

    logger.info("Orderbook saved to: " + output_file)


if __name__ == "__main__":
    fileConfig('config/logging_config.ini')
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Analyse level III order book data')
    parser.add_argument('--config', metavar='path', type=str, nargs='?',
                        help='path to config file')
    parser.add_argument('--start_time', metavar='HH-MM-DDTHH:MM:SS', type=str, nargs='?',
                        help='start time for simulation (overrides parameter in .ini file)')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    real_root = config['paths']['real_root']
    sim_root = config['paths']['sim_root']
    graphs_root = config['paths']['graphs_root']
    params_path = config['paths']['params_path']

    sampling_window = int(config['window']['sampling'])
    simulation_window = int(config['window']['simulation'])
    orderbook_window = int(config['window']['orderbook'])

    start_time = config['data']['start_time']

    product = config['data']['product']

    mode = config['behaviour']['mode']
    show_graphs = config['behaviour'].getboolean('show_graphs')
    save_graphs = config['behaviour'].getboolean('save_graphs')
    run_simulation = config['behaviour'].getboolean('run_simulation')
    fit_distributions = config['behaviour'].getboolean('fit_distributions')

    if args.start_time:
        logger.info("Using command line start_time")
        start_time = args.start_time

    start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")

    graphs = save_graphs or show_graphs

    print(graphs)
    print(run_simulation)
    print(fit_distributions)

    print(mode)

    if mode == "combined":
        combined_mode(start_time)
    elif mode == "multi-combined":
        multi_combined_mode(start_time)
    elif mode == "real":
        real_mode(start_time)
    elif mode == "simulation":
        simulation_mode()
    elif mode == "orderbook":
        orderbook_mode()


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    logger.debug("number of sell side interactions: " + str(num_sells))
    logger.debug("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells
