import argparse
import configparser
import datetime
import logging
import pathlib
import time

import dask.dataframe as dd

from logging.config import fileConfig

import pebble

from analysis.combined_analysis import CombinedAnalysis
from analysis.real_analysis import RealAnalysis
from analysis.simulation_analysis import SimulationAnalysis
from data_loader import DataLoader
from data_splitter import DataSplitter
from orderbook import OrderBook

from operator import itemgetter

from pympler import tracker


def get_all_data(st: datetime, config):
    root = config['full_paths']['root']

    real_root = root + config['part_paths']['real_root']
    product = config['data']['product']

    ob_w = int(config['window']['orderbook'])
    sam_w = int(config['window']['sampling'])
    sim_w = int(config['window']['simulation'])

    # Get all data which we will use to reconstruct the order book
    all_ob_start_time = st - datetime.timedelta(seconds=ob_w)
    all_ob_end_time = st
    all_ob_data = DataLoader().load_split_data(real_root, all_ob_start_time, all_ob_end_time, product)

    # Assume orderbook_window > sampling_window, and therefore filter already loaded ob data
    all_sample_start_time = st - datetime.timedelta(seconds=sam_w)
    all_sample_end_time = st
    all_sampling_data = map(lambda x: DataSplitter.get_between(x, all_sample_start_time, all_sample_end_time),
                            all_ob_data)

    # Get future data
    all_future_data_start_time = st
    all_future_data_end_time = st + datetime.timedelta(seconds=sim_w)
    all_future_data = DataLoader().load_split_data(real_root, all_future_data_start_time, all_future_data_end_time,
                                                   product)

    return all_ob_data, all_sampling_data, all_future_data


def combined_mode(st: datetime.datetime = None):
    if not st:
        st = datetime.datetime.strptime(config['data']['start_time'], "%Y-%m-%dT%H:%M:%S")
    run_simulation = config['behaviour'].getboolean('run_simulation')

    all_ob_data, all_sampling_data, all_future_data = get_all_data(st, config)

    combined_analysis = CombinedAnalysis(config, st, all_ob_data, all_sampling_data, all_future_data)

    if run_simulation:
        combined_analysis.run_simulation()
        future = combined_analysis.validate_analyses(prog_start)
        # Block until validation is done
        future.result()
        # if graphs:
        #     combined_analysis.print_stat_comparison()
        #     combined_analysis.graph_real_prices_with_simulated_confidence_intervals()


def add_secs(dt: datetime, secs: int):
    return dt + datetime.timedelta(seconds=secs)


def take_secs(dt: datetime, secs: int):
    return dt - datetime.timedelta(seconds=secs)


def multi_combined_mode(st: datetime.datetime = None):
    if not st:
        st = datetime.datetime.strptime(config['data']['start_time'], "%Y-%m-%dT%H:%M:%S")

    real_root = root_path + config['part_paths']['real_root']
    product = config['data']['product']

    ob_w = int(config['window']['orderbook'])
    sam_w = int(config['window']['sampling'])
    sim_w = int(config['window']['simulation'])

    num_predictions = int(config['data']['num_predictions'])
    interval = int(config['data']['interval'])

    all_data_st = take_secs(st, ob_w)
    all_data_et = add_secs(st, (num_predictions - 1) * interval)

    mem = tracker.SummaryTracker()

    all_data = DataLoader.load_split_data(real_root, all_data_st, all_data_et, product)

    for i in range(0, num_predictions):
        logger.info("Iteration " + str(i))
        sim_st = add_secs(st, interval * i)
        sim_et = add_secs(sim_st, sim_w)

        ob_st = take_secs(sim_st, ob_w)
        ob_et = sim_st

        sam_st = take_secs(sim_st, sam_w)
        sam_et = sim_st

        try:
            logger.info("Gathering data for simulation at: " + sim_st.isoformat())
            all_ob_data = map(lambda x: DataSplitter.get_between(x, ob_st, ob_et),
                              all_data)

            all_sampling_data = map(lambda x: DataSplitter.get_between(x, sam_st, sam_et),
                                    all_data)

            all_future_data = map(lambda x: DataSplitter.get_between(x, sim_st, sim_et),
                                  all_data)

            combined_analysis = CombinedAnalysis(config, sim_st, all_ob_data, all_sampling_data, all_future_data)

            combined_analysis.run_simulation()

            logger.info("Starting validation in other proc")
            combined_analysis.validate_analyses(prog_start)
            logger.info("Validation started")

            # Check that previous validation has ended
            # if validation_process is not None:
            #     validation_process.join()
            # logger.info("Starting validation in separate process")
            # validation_process = Process(target=combined_analysis.validate_analyses())
            # validation_process.start()
            # logger.info("Validation started")
            # validation_process.join()
        except Exception as exception:
            logger.error("Combined failed, skipping, at: " + sim_st.isoformat() + "\nError was\n" + str(exception))
        finally:
            print(sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10])


def real_mode(st: datetime.datetime = None):
    if not st:
        st = datetime.datetime.strptime(config['data']['start_time'], "%Y-%m-%dT%H:%M:%S")

    real_root = config['paths']['real_root']
    params_path = config['paths']['params_path']
    product = config['data']['product']
    sampling_window = int(config['window']['sampling'])
    fit_distributions = config['behaviour']['fit_distributions']
    graphs_root = config['paths']['graphs_root']

    sampling_window_start_time = st - datetime.timedelta(seconds=sampling_window)
    sampling_window_end_time = st
    orders_df, trades_df, cancels_df = DataLoader.load_split_data(real_root, sampling_window_start_time,
                                                                  sampling_window_end_time, product)
    real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Combined " + product)

    if fit_distributions and params_path:
        params = real_analysis.generate_order_params()
        real_analysis.json_to_file(params, params_path)
    if graphs_root:
        real_analysis.generate_graphs(graphs_root)


def simulation_mode(st: datetime.datetime = None):
    if not st:
        st = datetime.datetime.strptime(config['data']['start_time'], "%Y-%m-%dT%H:%M:%S")

    SimulationAnalysis(config, st).analyse()


def orderbook_mode(st: datetime.datetime = None):
    if not st:
        st = datetime.datetime.strptime(config['data']['start_time'], "%Y-%m-%dT%H:%M:%S")

    root = config['full_paths']['root']
    orderbook_path = root + config['part_paths']['orderbook_path']

    orderbook_window_start_time = st - datetime.timedelta(seconds=orderbook_window)
    orderbook_window_end_time = st

    orders_df, trades_df, cancels_df = DataLoader.load_split_data(real_root, orderbook_window_start_time,
                                                                  orderbook_window_end_time, product)

    ob_state = OrderBook.load_orderbook_state(orderbook_path)
    orderbook = OrderBook.get_orderbook(orders_df, trades_df, cancels_df, ob_state)
    output_file = "/Users/jamesprince/project-data/orderbook-" + orderbook_window_end_time.isoformat() + ".csv"
    OrderBook.orderbook_to_file(orderbook, output_file)

    logger.info("Orderbook saved to: " + output_file)


if __name__ == "__main__":
    t0 = time.time()

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

    mode = config['behaviour']['mode']
    root_path = config['full_paths']['root']

    prog_start = datetime.datetime.now()

    # Ensure all paths exist
    for path_key in config['part_paths']:
        try:
            if pathlib.Path(root_path + config['part_paths'][path_key]).mkdir(parents=True, exist_ok=True):
                raise FileNotFoundError
        except FileExistsError:
            pass
        except FileNotFoundError:
            logger.info(config['part_paths'][path_key] + " does not exist")
        logger.info(config['part_paths'][path_key] + " exists")

    if mode == "combined":
        combined_mode()
    elif mode == "multi-combined":
        multi_combined_mode()
    elif mode == "real":
        real_mode()
    elif mode == "simulation":
        simulation_mode()
    elif mode == "orderbook":
        orderbook_mode()

    logger.info("Program took " + str(time.time() - t0) + " seconds")


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    logger.debug("number of sell side interactions: " + str(num_sells))
    logger.debug("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells
