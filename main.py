import argparse
import configparser
import datetime
import logging
import pathlib
import time
from logging.config import fileConfig
from operator import itemgetter

import dask.dataframe as dd
import matplotlib
from pympler import tracker

from modes.backtest import Backtest
from modes.real_analysis import RealAnalysis
from modes.simulation_analysis import SimulationAnalysis
from config import Config
from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from orderbook import OrderBook
from output.writer import Writer


def get_all_data(st: datetime, config):
    # Get all data which we will use to reconstruct the order book
    all_ob_start_time = st - datetime.timedelta(seconds=config.orderbook_window)
    all_ob_end_time = st
    all_ob_data = DataLoader().load_split_data(config.real_root, all_ob_start_time, all_ob_end_time, config.product)

    # Assume orderbook_window > sampling_window, and therefore filter already loaded ob data
    all_sample_start_time = st - datetime.timedelta(seconds=config.sampling_window)
    all_sample_end_time = st
    all_sampling_data = map(lambda x: DataSplitter.get_between(x, all_sample_start_time, all_sample_end_time),
                            all_ob_data)

    # Get future data
    all_future_data_start_time = st
    all_future_data_end_time = st + datetime.timedelta(seconds=config.sampling_window)
    all_future_data = DataLoader().load_split_data(config.real_root, all_future_data_start_time,
                                                   all_future_data_end_time, config.product)

    return all_ob_data, all_sampling_data, all_future_data


def add_secs(dt: datetime, secs: int):
    return dt + datetime.timedelta(seconds=secs)


def take_secs(dt: datetime, secs: int):
    return dt - datetime.timedelta(seconds=secs)


def backtest_mode(st: datetime.datetime = None):
    all_data_st = take_secs(st, max(config.orderbook_window, config.sampling_window))
    all_data_et = add_secs(st, config.num_predictions * config.interval)

    mem = tracker.SummaryTracker()

    all_data = DataLoader.load_split_data(config.real_root, all_data_st, all_data_et, config.product)

    validate = None

    for i in range(0, config.num_predictions):
        logger.info("Iteration " + str(i))
        sim_st = add_secs(st, config.interval * i)
        sim_et = add_secs(sim_st, config.simulation_window)

        ob_st = take_secs(sim_st, config.orderbook_window)
        ob_et = sim_st

        sam_st = take_secs(sim_st, config.sampling_window)
        sam_et = sim_st

        success = False
        backtest = None

        try:
            logger.info("Gathering data for simulation at: " + sim_st.isoformat())
            all_ob_data = map(lambda x: DataSplitter.get_between(x, ob_st, ob_et),
                              all_data)

            all_sampling_data = map(lambda x: DataSplitter.get_between(x, sam_st, sam_et),
                                    all_data)

            all_future_data = map(lambda x: DataSplitter.get_between(x, sim_st, sim_et),
                                  all_data)

            backtest = Backtest(config, sim_st, all_ob_data, all_sampling_data, all_future_data)

        except Exception as e:
            logger.error("Error occurred when gathering data: " + str(e))
            backtest = None

        if validate is not None:
            validate.result()

        try:
            if backtest is not None:
                success = backtest.prepare_simulation()

        except Exception as e:
            logger.error("Simulation preparation failed, skipping, at: " + sim_st.isoformat() + "\nError was\n" + str(e))
            continue

        try:
            if backtest is not None:
                backtest.run_simulation()
        except Exception as e:
            logger.error(
                "Simulation failed, skipping, at: " + sim_st.isoformat() + "\nError was\n" + str(e))
            continue
        finally:
            print(sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10])
            if success:
                logger.info("Starting validation in other proc")
                validate = backtest.validate_analyses(prog_start)
                logger.info("Validation started")

    validate.result()


def real_mode(st: datetime.datetime = None):
    if not st:
        st = datetime.datetime.strptime(config['data']['start_time'], "%Y-%m-%dT%H:%M:%S")

    sampling_window_start_time = st - datetime.timedelta(seconds=config.sampling_window)
    sampling_window_end_time = st
    orders_df, trades_df, cancels_df = DataLoader.load_split_data(config.real_root, sampling_window_start_time,
                                                                  sampling_window_end_time, config.product)
    real_analysis = RealAnalysis(orders_df, trades_df, cancels_df, "Backtest " + config.product)

    if config.fit_distributions and config.params_path:
        params = generate_order_params(real_analysis.trades_df, real_analysis.orders_df, real_analysis.cancels_df)
        Writer.json_to_file(params, config.params_path)
    if config.graphs_root:
        real_analysis.generate_graphs(config.graphs_root)


def simulation_mode(st: datetime.datetime = None):
    SimulationAnalysis(config, st).analyse()


def orderbook_mode(st: datetime.datetime = None):
    closest_ob_state, closest_ob_state_str = OrderBook.locate_closest_ob_state(config.orderbook_output_root, st)
    orders_df, trades_df, cancels_df = DataLoader.load_split_data(config.real_root, closest_ob_state,
                                                                  st, config.product)
    ob_state_path = config.root_path + closest_ob_state_str
    ob_state = OrderBook.load_orderbook_state(ob_state_path)
    orderbook = OrderBook.get_orderbook(orders_df, trades_df, cancels_df, ob_state)
    output_file = "/Users/jamesprince/project-data/orderbook-" + st.isoformat() + ".csv"
    OrderBook.orderbook_to_file(orderbook, output_file)

    logger.info("Orderbook saved to: " + output_file)


if __name__ == "__main__":
    t0 = time.time()

    fileConfig('config/logging_config.ini')
    logger = logging.getLogger("Main")

    parser = argparse.ArgumentParser(description='Analyse level III order book data')
    parser.add_argument('--config', metavar='path', type=str, nargs='?',
                        help='path to config file')
    parser.add_argument('--start_time', metavar='HH-MM-DDTHH:MM:SS', type=str, nargs='?',
                        help='start time for simulation (overrides parameter in .ini file)')
    args = parser.parse_args()

    conf = configparser.ConfigParser()
    conf.read(args.config)
    config = Config(conf)

    prog_start = datetime.datetime.now()

    # Ensure all paths exist
    for path_key in conf['part_paths']:
        try:
            if pathlib.Path(config.root_path + conf['part_paths'][path_key]).mkdir(parents=True, exist_ok=True):
                raise FileNotFoundError
        except FileExistsError:
            pass
        except FileNotFoundError:
            logger.info(conf['part_paths'][path_key] + " does not exist")
        logger.info(conf['part_paths'][path_key] + " exists")

    if config.mode == "backtest":
        backtest_mode(config.start_time)
    elif config.mode == "real":
        real_mode()
    elif config.mode == "simulation":
        simulation_mode()
    elif config.mode == "orderbook":
        orderbook_mode()

    logger.info("Program took " + str(time.time() - t0) + " seconds")


def sides(df: dd) -> (int, int):
    num_buys = len(df[df['side'] == 'sell'])
    num_sells = len(df[df['side'] == 'buy'])
    logger.debug("number of sell side interactions: " + str(num_sells))
    logger.debug("number of buy side interactions: " + str(num_buys))
    return num_buys, num_sells
