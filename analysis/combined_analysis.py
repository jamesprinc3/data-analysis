from analysis.simulation_analysis import SimulationAnalysis
from analysis.real_analysis import RealAnalysis
from data_loader import DataLoader
from data_utils import DataUtils
from stats import Statistics
import matplotlib.pyplot as plt
import numpy as np


class CombinedAnalysis:
    def __init__(self, sim_root, real_root):
        self.sim_analysis = SimulationAnalysis(sim_root, "Combined Analysis")

        self.real_root = real_root
        pass

    def graph_real_prices_with_simulated_confidence_intervals(self):
        interval = 10  # seconds
        limit = 300
        times = list(range(interval, limit, interval))
        confidence_intervals = self.sim_analysis.calculate_confidence_at_times(times)
        print(confidence_intervals)

        # Nabbed from: https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/

        def plot_mean_and_CI_and_real_values(mean, lb, ub, times, real_times, real_prices, color_mean=None,
                                             color_shading=None):
            # plot the shaded range of the confidence intervals
            plt.figure(figsize=(12, 8))
            plt.ylim(8000, 9000)
            plt.title("BTC-USD Price prediction")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Price ($)")
            plt.fill_between(times, ub, lb,
                             color=color_shading, alpha=.5)
            # plot the mean on top
            plt.plot(times, mean, color_mean, label="Simulated")
            plt.plot(real_times, real_prices, 'r+', label="Real")
            plt.legend()

        df = DataLoader().load_real_data(self.real_root)[['time', 'price', 'reason']].compute()
        trades_df = Statistics().get_trades(df)

        start_price = trades_df.dropna().iloc[0]['price']

        trades_df['time'] = DataUtils().get_times_in_seconds_after_start(trades_df['time'])
        real_times = trades_df['time']
        real_prices = trades_df['price']

        # generate 3 sets of random means and confidence intervals to plot
        mean10 = [start_price] + list(map(lambda lu: (lu[0] + lu[1]) / 2, confidence_intervals.values()))
        print(mean10)
        mean0 = np.array(mean10)
        ub0 = np.array([start_price] + list(map(lambda x: x[1], confidence_intervals.values())))
        lb0 = np.array([start_price] + list(map(lambda x: x[0], confidence_intervals.values())))

        times = [0] + times

        # plot the data
        fig = plt.figure(1, figsize=(7, 2.5))
        plot_mean_and_CI_and_real_values(mean0, ub0, lb0, times, real_times, real_prices, color_mean='k',
                                         color_shading='k')

        plt.show()
