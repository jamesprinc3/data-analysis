import configparser
from datetime import timedelta
from unittest import TestCase

import pandas as pd

from backtest_config import BacktestConfig
from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from modes.evaluation import Evaluation


class TestEvaluation(TestCase):
    def setUp(self):
        self.root = "/Users/jamesprince/project-data/"
        self.corr_root = self.root + "results/correlations/"
        self.sim_root = self.root + "results/sims/LTC-USD/"
        self.real_root = self.root + "data/consolidated-feed/LTC-USD/"

        conf = configparser.ConfigParser()
        conf.read("../config/backtest.ini")
        self.config = BacktestConfig(conf)

        self.sim_st = self.config.start_time

    def test_compare_order_metrics(self):
        sim_root = self.config.sim_root + self.sim_st.date().isoformat() + "/" + self.sim_st.time().isoformat() + "/"
        all_sims = DataLoader().load_sim_data(sim_root)
        all_sim_limit_orders = list(map(lambda sim: DataSplitter.get_limit_orders(sim[0].compute()), all_sims))
        all_sim_market_orders = list(map(lambda sim: DataSplitter.get_market_orders(sim[0].compute()), all_sims))
        all_sim_trades = list(map(lambda sim: sim[1].compute(), all_sims))
        all_sim_cancels = list(map(lambda sim: sim[2].compute(), all_sims))

        feed_df = DataLoader().load_feed(self.config.real_root, self.sim_st,
                                         self.sim_st + timedelta(seconds=self.config.simulation_window),
                                         self.config.product)
        real_orders = DataSplitter.get_orders(feed_df)
        real_limit_orders = DataSplitter.get_limit_orders(real_orders)
        real_market_orders = DataSplitter.get_market_orders(real_orders)
        real_trades = DataSplitter.get_trades(feed_df)
        real_trades['size'] = pd.to_numeric(real_trades['remaining_size'])
        real_cancels = DataSplitter.get_cancellations(feed_df)
        real_cancels['size'] = pd.to_numeric(real_cancels['remaining_size'])

        print("Order Buy/Sell limit metrics")
        Evaluation.compare_order_metrics(real_limit_orders, all_sim_limit_orders)
        print("Order Buy/Sell market metrics")
        Evaluation.compare_order_metrics(real_market_orders, all_sim_market_orders)
        print("Cancel metrics")
        Evaluation.compare_order_metrics(real_cancels, all_sim_cancels)
        print("Trade metrics")
        Evaluation.compare_metrics(real_trades, all_sim_trades)
    
    def test_can_correlate_10(self):
        df = Evaluation.load_csv(self.corr_root + "LTC-cov.csv")

        Evaluation.compare_returns(df)

    def test_can_correlate_50(self):
        df = Evaluation.load_csv(self.corr_root + "LTC-USD-50-sims.csv")

        Evaluation.compare_returns(df)

    def test_correlate_100_percentiles(self):
        df = Evaluation.load_csv(self.corr_root + "LTC-USD-100-percentiles.csv")

        Evaluation.compare_returns(df)

    def test_correlate_100_percentiles_midprice(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-100-percentiles-midprice.csv")

        Evaluation.compare_returns(df)

    def test_correlate_100_percentiles_midprice_fix(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-100-fix-midprices.csv")

        Evaluation.compare_returns(df)

    def test_correlate_100_percentiles_trades_fix(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-100-fix-trades.csv")

        Evaluation.compare_returns(df)

    def test_correlate_100_percentiles_midprice_inv(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-100-inv-midprice.csv")

        Evaluation.compare_returns(df, compare_lyapunov_exponent=True)

    def test_correlate_100_percentiles_trades_inv(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-100-inv-trade.csv")

        Evaluation.compare_returns(df)

    def test_correlate_utc_midprice(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-utc.csv")

        Evaluation.compare_returns(df)

    def test_correlate_utc_mid_midprices(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-mid-midprices.csv")

        Evaluation.compare_returns(df)

    def test_correlate_utc_mid_trade(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD-mid-trade.csv")

        Evaluation.compare_returns(df)

    def test_correlate_report_LTC_USD(self):
        df = Evaluation.load_csv(
            self.corr_root + "report/LTC-USD-100sims-17-5-2018.csv")

        Evaluation.compare_returns(df)

    def test_correlate_report_LTC_USD_some_removed(self):
        df = Evaluation.load_csv(
            self.corr_root + "report/LTC-USD-sims-some-removed-17-5-2018.csv")

        Evaluation.compare_returns(df)

    def test_correlate_report_LTC_USD_2_day(self):
        df = Evaluation.load_csv(
            self.corr_root + "report/LTC-USD-17-18-5-2018.csv")

        Evaluation.compare_returns(df)

    def test_correlate_cancel_relative(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD/cancel-relative-midprice.csv")

        Evaluation.compare_returns(df)

    def test_correlate_ETH_USD(self):
        df = Evaluation.load_csv(
            self.corr_root + "ETH-USD/17-05-18-midprice.csv")

        Evaluation.compare_returns(df, window=100)

    def test_correlate_ETH_USD_2(self):
        df = Evaluation.load_csv(
            self.corr_root + "ETH-USD/17-05-2018-2-midprice.csv")

        Evaluation.compare_returns(df, window=100)

    def test_correlate_ETH_USD_all(self):
        df = Evaluation.load_csv(
            self.corr_root + "ETH-USD/17-05-2018-all-midprice.csv")

        Evaluation.compare_returns(df, window=100)

    def test_correlate_ETH_USD_all_trade(self):
        df = Evaluation.load_csv(
            self.corr_root + "ETH-USD/17-05-2018-all-trade.csv")

        Evaluation.compare_returns(df, window=100)

    def test_correlate_BCH_USD_midprice(self):
        df = Evaluation.load_csv(
            self.corr_root + "BCH-USD/17-05-2018-midprice.csv")

        Evaluation.compare_returns(df, window=100)

    def test_correlate_BCH_USD_trade(self):
        df = Evaluation.load_csv(
            self.corr_root + "BCH-USD/17-05-2018-trade.csv")

        Evaluation.compare_returns(df, window=100)

    def test_correlate_LTC_USD_26_05_2018_100(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD/26-05-2018-100-midprice.csv")

        Evaluation.compare_returns(df)

    def test_correlate_LTC_USD_26_05_2018_200(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD/26-05-2018-200-midprice.csv")

        Evaluation.compare_returns(df)

    def test_correlate_LTC_USD_26_05_2018_all(self):
        df = Evaluation.load_csv(
            self.corr_root + "LTC-USD/26-05-2018-all-midprice.csv")

        Evaluation.compare_returns(df)


    def test_hurst_regions(self):
        one_nine = Evaluation.load_csv(
            self.corr_root + "1-9.csv")

        nine_seventeen = Evaluation.load_csv(
            self.corr_root + "9-17.csv")

        seventeen_twenty_two = Evaluation.load_csv(
            self.corr_root + "17-22.csv")

        Evaluation.compare_returns(one_nine.append(seventeen_twenty_two))
        # Evaluation.correlate(nine_seventeen)
        # Evaluation.correlate(seventeen_twenty_two)
