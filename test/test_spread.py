from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd

from data.data_loader import DataLoader


class TestSpread(TestCase):
    def test_plot_monte(self):
        root = "/Users/jamesprince/project-data/random-walk/"
        files = DataLoader.get_files_in_dir(root)

        for file in files:
            sim = pd.read_csv(root + file)

            plt.plot(sim['time'], sim['price'])

        plt.show()

    def test_plot_confidence(self):
        root = "/Users/jamesprince/project-data/random-walk/"
        files = DataLoader.get_files_in_dir(root)

        time_steps = 100
        data = {}
        for i in range(0, time_steps):
            data[i] = []

        for file in files:
            sim = pd.read_csv(root + file)

            for i in range(0, time_steps):
                data[i].append(sim['spread'].iloc[i])

        TestSpread.plot_confidence(data)

        plt.show()

    @staticmethod
    def plot_confidence(data):
        for k, v in data.items():
            lb, ub = TestSpread.get_confidence(v)
            plt.plot(k, lb, 'r+')
            plt.plot(k, (ub + lb) / 2, 'g+')
            plt.plot(k, ub, 'b+')

    @staticmethod
    def get_confidence(a, level=0.95):
        import numpy as np, scipy.stats as st
        return st.t.interval(level, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
