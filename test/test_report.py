from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


class TestReport(TestCase):

    def test_get_binom(self):

        n = 100
        p = 0.5

        plt.figure(figsize=(12, 8))
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, n)
        ax.plot(x, binom.pmf(x, n, p), ms=8)
        ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
        plt.show()


    def test_get_binom_coloured(self):

        k = 58
        n = 100
        p = 0.5

        plt.figure(figsize=(12, 8))
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0, k)
        y = np.arange(k, n)
        ax.plot(x, binom.pmf(x, n, p), ms=8)
        ax.plot(y, binom.pmf(y, n, p), ms=8)
        ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
        ax.vlines(y, 0, binom.pmf(y, n, p), colors='r', lw=5, alpha=0.5)
        plt.show()

        plt.savefig("binom.png")

    def test_get_example_cdf(self):
        plt.figure(figsize=(12, 8))
        plt.plot([0, 1, 1, 2, 2, 3, 3], [0, 0, 0.25, 0.25, 0.6, 0.6, 1], 'b', label="CDF")
        plt.plot([0, 2, 2], [0.41, 0.41, 0], 'r', linestyle=':', label="Sample")
        plt.legend()

        plt.show()
