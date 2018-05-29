import datetime
from unittest import TestCase
from data.data_loader import DataLoader

import pandas as pd
import numpy as np

from modes.evaluation import Evaluation


class TestCorrelation(TestCase):

    def test_can_load_data(self):
        rp, sp = Evaluation.load("/Users/jamesprince/project-data/results/correlations/2018-05-23T15:15:12.267315.csv")

    def test_can_correlate(self):
        df = Evaluation.load("/Users/jamesprince/project-data/results/correlations/LTC-USD.csv")
        # df = Correlation.load("/Users/jamesprince/project-data/results/correlations/2018-05-22T10:57:45.196921.csv")

        Evaluation.correlate(df)
