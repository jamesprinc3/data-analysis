from unittest import TestCase

from modes.evaluation import Evaluation


class TestCorrelation(TestCase):
    def test_can_correlate_10(self):
        df = Evaluation.load("/Users/jamesprince/project-data/results/correlations/LTC-cov.csv")

        Evaluation.correlate(df)

    def test_can_correlate_50(self):
        df = Evaluation.load("/Users/jamesprince/project-data/results/correlations/LTC-USD-50-sims.csv")

        Evaluation.correlate(df)

