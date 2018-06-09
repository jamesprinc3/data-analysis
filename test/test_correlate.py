from unittest import TestCase

from modes.evaluation import Evaluation


class TestCorrelation(TestCase):
    def setUp(self):
        self.root = "/Users/jamesprince/project-data/results/correlations/"
    
    def test_can_correlate_10(self):
        df = Evaluation.load(self.root + "LTC-cov.csv")

        Evaluation.correlate(df)

    def test_can_correlate_50(self):
        df = Evaluation.load(self.root + "LTC-USD-50-sims.csv")

        Evaluation.correlate(df)

    def test_correlate_100_percentiles(self):
        df = Evaluation.load(self.root + "LTC-USD-100-percentiles.csv")

        Evaluation.correlate(df)

    def test_correlate_100_percentiles_midprice(self):
        df = Evaluation.load(
            self.root + "LTC-USD-100-percentiles-midprice.csv")

        Evaluation.correlate(df)

    def test_correlate_100_percentiles_midprice_fix(self):
        df = Evaluation.load(
            self.root + "LTC-USD-100-fix-midprices.csv")

        Evaluation.correlate(df)

    def test_correlate_100_percentiles_trades_fix(self):
        df = Evaluation.load(
            self.root + "LTC-USD-100-fix-trades.csv")

        Evaluation.correlate(df)

    def test_correlate_100_percentiles_midprice_inv(self):
        df = Evaluation.load(
            self.root + "LTC-USD-100-inv-midprice.csv")

        Evaluation.correlate(df, compare_lyapunov_exponent=True)

    def test_correlate_100_percentiles_trades_inv(self):
        df = Evaluation.load(
            self.root + "LTC-USD-100-inv-trade.csv")

        Evaluation.correlate(df)

    def test_hurst_regions(self):
        one_nine = Evaluation.load(
            self.root + "1-9.csv")

        nine_seventeen = Evaluation.load(
            self.root + "9-17.csv")

        seventeen_twenty_two = Evaluation.load(
            self.root + "17-22.csv")

        Evaluation.correlate(one_nine.append(seventeen_twenty_two))
        # Evaluation.correlate(nine_seventeen)
        # Evaluation.correlate(seventeen_twenty_two)
