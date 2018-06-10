import configparser
from datetime import timedelta
from unittest import TestCase

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
        all_sim_orders = list(map(lambda sim: sim[0].compute(), all_sims))
        all_sim_trades = list(map(lambda sim: sim[1].compute(), all_sims))
        all_sim_cancels = list(map(lambda sim: sim[2].compute(), all_sims))

        feed_df = DataLoader().load_feed(self.config.real_root, self.sim_st,
                                         self.sim_st + timedelta(seconds=self.config.simulation_window),
                                         self.config.product)
        real_orders = DataSplitter.get_orders(feed_df)
        real_trades = DataSplitter.get_trades(feed_df)
        real_cancels = DataSplitter.get_cancellations(feed_df)

        print("Order metrics")
        Evaluation.compare_metrics(real_orders, all_sim_orders)
        print("Order Buy/Sell metrics")
        Evaluation.compare_order_metrics(real_orders, all_sim_orders)
        print("Trade metrics")
        Evaluation.compare_metrics(real_trades, all_sim_trades)
        print("Cancel metrics")
        Evaluation.compare_metrics(real_cancels, all_sim_cancels)
    
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
