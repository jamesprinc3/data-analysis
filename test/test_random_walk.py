from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import DataLoader


class TestRandomWalk(TestCase):
    def test_plot_monte(self):
        root = "/Users/jamesprince/project-data/random-walk/"
        files = DataLoader.get_files_in_dir(root)

        for file in files:
            sim = pd.read_csv(root + file)

            plt.plot(sim['time'], sim['price'])

        plt.show()

    def test_plot_percentiles(self):
        data = self.get_data()

        for k, v in data.items():
            lower, upper = TestRandomWalk.get_percentiles(v)
            plt.plot(k, lower, 'r+')
            plt.plot(k, np.mean(v), 'g+')
            plt.plot(k, upper, 'b+')

        plt.show()

    def get_data(self):
        root = "/Users/jamesprince/project-data/random-walk/"
        files = DataLoader.get_files_in_dir(root)
        time_steps = 100
        data = {}
        for i in range(0, time_steps):
            data[i] = []
        for file in files:
            sim = pd.read_csv(root + file)

            for i in range(0, time_steps):
                data[i].append(sim['price'].iloc[i])
        return data

    def test_plot_histo(self):
        data = self.get_data()

        # for step_index, prices in data.items():
        #     self.plot_histo(prices, step_index)

        self.plot_histo(data[99], 99)

    def plot_histo(self, prices, step_index):
        plt.hist(prices, bins=20, density=True, alpha=0.5)
        plt.title("Step " + str(step_index))
        plt.show()

    # Number of samples is very high, so we can use t=1.96 for 95% conf. interval
    @staticmethod
    def get_confidence(data, t=1.96):
        mean = np.mean(data)
        std_err = np.std(data)
        n = len(data)

        ub = mean + (t * (std_err / np.sqrt(n)))
        lb = mean - (t * (std_err / np.sqrt(n)))

        return lb, ub

    @staticmethod
    def get_percentiles(data, level=0.025):
        n = len(data)
        data.sort()

        print(data)

        lower = data[int(n * level)]
        upper = data[int(n * (1 - level))]

        return lower, upper

    # @staticmethod
    # def get_confidence(a, level=0.95):
    #     import numpy as np, scipy.stats as st
    #     return st.t.interval(level, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
