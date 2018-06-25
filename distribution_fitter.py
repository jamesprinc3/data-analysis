# Some methods in this file were inspired by: https://stackoverflow.com/a/37616966

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st


class DistributionFitter:
    def __init__(self, config):
        self.config = config
        matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
        matplotlib.style.use('ggplot')



    @staticmethod
    def make_pdf(dist, params, size=10000):
        """Generate distributions' Probability Distribution Function"""

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf

    @staticmethod
    def plot_data_with_distribution(data, dist, fit_params, data_desc: str, xlabel: str, bins=200, show=False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        pdf = DistributionFitter.make_pdf(dist, fit_params)
        ax = pdf.plot(lw=2, label='PDF', legend=True)
        data.plot(kind='hist', bins=bins, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

        ax.set_title(data_desc)
        ax.set_xlabel(xlabel)

        if show:
            plt.show()

    @staticmethod
    def get_distribution_string(best_fit, best_fit_params):
        # Extract name of best fit distribution
        best_dist = getattr(st, best_fit.name)

        # Format the name
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit.name, param_str)

        return best_dist, dist_str
