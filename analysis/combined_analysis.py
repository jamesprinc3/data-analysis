from analysis.simulation_analysis import SimulationAnalysis
from analysis.real_analysis import RealAnalysis
from data_loader import DataLoader
from data_utils import DataUtils
from stats import Statistics

class CombinedAnalysis:

    def __init__(self, sim_root, real_root):
        self.sim_analysis = SimulationAnalysis(sim_root, "Combined Analysis")

        self.real_root = real_root
        pass

    def graph_real_prices_with_simulated_confidence_intervals(self):
        times = [10, 20, 30, 40, 50, 60]
        confidence_intervals = self.sim_analysis.calculate_confidence_at_times(times)
        print(confidence_intervals)

        #Nabbed from: https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import colorConverter as cc
        import numpy as np

        def plot_mean_and_CI_and_real_values(mean, lb, ub, times, real_times, real_prices,  color_mean=None, color_shading=None):
            # plot the shaded range of the confidence intervals
            plt.figure(figsize=(12, 8))
            plt.ylim(8000, 9000)
            # print("mean.shape[0] ")
            # print(mean.shape[0])
            plt.fill_between(times, ub, lb,
                             color=color_shading, alpha=.5)
            # plot the mean on top
            plt.plot(times, mean, color_mean)
            plt.plot(real_times, real_prices, 'r+')

        # mean1 = np.random.random(50) + 2
        # ub1 = mean1 + np.random.random(50) + .5
        # lb1 = mean1 - np.random.random(50) - .5

        # generate 3 sets of random means and confidence intervals to plot
        mean10 = list(map(lambda lu: (lu[0] + lu[1]) / 2, confidence_intervals.values()))
        print(mean10)
        mean0 = np.array(mean10)
        ub0 = np.array(list(map(lambda x: x[1], confidence_intervals.values())))
        lb0 = np.array(list(map(lambda x: x[0], confidence_intervals.values())))

        df = DataLoader().load_real_data(self.real_root)[['time', 'price', 'reason']].compute()
        trades_df = Statistics().get_trades(df)
        trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
        real_times = trades_df['time']
        real_prices = trades_df['price']
        #
        # mean2 = np.random.random(50) - 1
        # ub2 = mean2 + np.random.random(50) + .5
        # lb2 = mean2 - np.random.random(50) - .5

        # plot the data
        fig = plt.figure(1, figsize=(7, 2.5))
        plot_mean_and_CI_and_real_values(mean0, ub0, lb0, times, real_times, real_prices, color_mean='k', color_shading='k')
        # plot_mean_and_CI(mean1, ub1, lb1, color_mean='b', color_shading='b')
        # plot_mean_and_CI(mean2, ub2, lb2, color_mean='g--', color_shading='g')

        plt.show()
