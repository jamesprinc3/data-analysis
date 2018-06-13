import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from prettytable import PrettyTable

from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from stats import Statistics


class Evaluation:

    @staticmethod
    def load_csv(path: str):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def get_plt(window):
        plt.figure(figsize=(12, 8))

        maximum, minimum = Evaluation.get_plot_min_max(window)
        # plt.xlim(minimum, maximum)
        # plt.ylim(minimum, maximum)

        plt.plot([minimum, maximum], [minimum, maximum])

        return plt

    @staticmethod
    def get_plot_min_max(window):
        minimum = -window / 2
        maximum = window / 2
        return maximum, minimum

    @staticmethod
    def get_direction(final_lower, final_upper, start_price):
        if final_lower > start_price:
            return 1
        elif final_upper < start_price:
            return -1
        else:
            return 0

    @staticmethod
    def compare_order_metrics(real_orders: pd.DataFrame, multi_sim_orders: List[pd.DataFrame]):
        """Compares metrics which only make sense in orders (e.g. buy/sell split)"""

        real_buy_orders = DataSplitter.get_side("buy", real_orders)
        sim_buy_orders = list(map(lambda sim: DataSplitter.get_side("buy", sim), multi_sim_orders))

        print("Buy metrics:")
        Evaluation.compare_metrics(real_buy_orders, sim_buy_orders)

        real_sell_orders = DataSplitter.get_side("sell", real_orders)
        sim_sell_orders = list(map(lambda sim: DataSplitter.get_side("sell", sim), multi_sim_orders))

        print("Sell metrics:")
        Evaluation.compare_metrics(real_sell_orders, sim_sell_orders)

    @staticmethod
    def compare_metrics(real_df: pd.DataFrame, multi_sim_feed: List[pd.DataFrame]):
        """Compare flows of categories of events (e.g. orders, cancels, trades) from the simulations vs the real orders"""

        sim_df = multi_sim_feed[0]
        for i in range(1, len(multi_sim_feed)):
            sim_df = sim_df.append(multi_sim_feed[i])

        t = PrettyTable(['Metric', 'Real', 'Sim'])

        real_order_count = len(real_df)
        avg_order_count = int(len(sim_df) / len(multi_sim_feed))
        t.add_row(['Count', real_order_count, avg_order_count])

        t = Evaluation.compare_col_func(Statistics.get_mean, "mean", "price", real_df, sim_df, t)
        t = Evaluation.compare_col_func(Statistics.get_std_dev, "std. dev. ", "price", real_df,
                                        sim_df.sample(avg_order_count), t)
        t = Evaluation.compare_col_func(Statistics.get_mean, "mean", "size", real_df, sim_df, t)
        t = Evaluation.compare_col_func(Statistics.get_std_dev, "std. dev. ", "size", real_df,
                                        sim_df.sample(avg_order_count), t)

        t = Evaluation.compare_col_func(lambda col_name, df: df[col_name].sum(), "volume", "size", real_df,
                                        sim_df.sample(avg_order_count), t)

        t.align = 'r'
        print(t)

    @staticmethod
    def compare_col_func(f: callable, data_desc: str, column: str, real_df: pd.DataFrame, sim_df: pd.DataFrame, t):
        """Compare sim and real data by calling a function on one of their common columns"""
        real_res = f(column, real_df)
        sim_res = f(column, sim_df)
        t.add_row([data_desc + " " + column, real_res, sim_res])

        return t

    @staticmethod
    def compare_returns(df, compare_hurst_exp: bool = False, compare_lyapunov_exponent=False, window: int = 10):
        """Compare returns of the simulated markets with the real market"""
        product = "LTC-USD"

        plt = Evaluation.get_plt(window)

        df = Evaluation.calculate_aux_rows(df, window)
        Evaluation.print_sim_summary(df)

        Evaluation.plot_linregress(df['rp_diff'], df['sp_diff'],
                                   window, xlabel="Real 5 minute returns",
                                   ylabel="Simulated 5 minute returns")

        if compare_lyapunov_exponent:
            step_minutes = 1
            window_minutes = 240

            st_str = df['start_time'].min()
            et_str = df['start_time'].max()

            st = datetime.datetime.strptime(st_str, "%Y-%m-%dT%H:%M:%S") + datetime.timedelta(
                minutes=window_minutes - 60)
            et = datetime.datetime.strptime(et_str, "%Y-%m-%dT%H:%M:%S") + datetime.timedelta(
                minutes=window_minutes - 60)
            _, trades, _ = DataLoader.load_split_data("/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/",
                                                      st,
                                                      et,
                                                      product)

            times, lyap_exps = Statistics.get_lyapunov_exponent_over_time(trades, st, et, step_minutes, window_minutes)
            lyap_df = pd.DataFrame({'start_time': list(map(lambda t: t.isoformat(), times)), 'lyap_exp': lyap_exps})

            joined_df = pd.merge(df, lyap_df, how='left', on='start_time')
            dropped_df = joined_df.dropna(subset=['lyap_exp'])

            x1 = dropped_df['lyap_exp']
            x2 = dropped_df['rp_sp_diff']

            plt.xlim(-0.2, 0.2)
            plt.ylim(-10, 10)

            Evaluation.plot_linregress(x1, x2,
                                       window,
                                       show_regression_line=True,
                                       xlabel="Lyapunov Exponent",
                                       ylabel="Diff between real and simulated returns")

        if compare_hurst_exp:
            st_str = df['start_time'].min()
            et_str = df['start_time'].max()

            step_minutes = 5
            window_minutes = 5

            st = datetime.datetime.strptime(st_str, "%Y-%m-%dT%H:%M:%S") - datetime.timedelta(minutes=window_minutes)
            et = datetime.datetime.strptime(et_str, "%Y-%m-%dT%H:%M:%S") - datetime.timedelta(minutes=window_minutes)
            _, trades, _ = DataLoader.load_split_data("/Users/jamesprince/project-data/data/consolidated-feed/LTC-USD/",
                                                      st,
                                                      et,
                                                      product)

            times, lyap_exps = Statistics.get_hurst_exponent_over_time(trades, st, et, step_minutes, window_minutes)
            lyap_df = pd.DataFrame({'start_time': list(map(lambda t: t.isoformat(), times)), 'hurst_exp': lyap_exps})

            print(lyap_df)

            joined_df = pd.merge(df, lyap_df, how='left', on='start_time')
            dropped_df = joined_df.dropna(subset=['hurst_exp'])

            x1 = dropped_df['hurst_exp']
            x2 = dropped_df['rp_sp_diff']

            Evaluation.plot_linregress(x1, x2,
                                       window,
                                       show_regression_line=True,
                                       xlabel="Hurst Exponent",
                                       ylabel="Diff between real and simulated returns")

    @staticmethod
    def calculate_aux_rows(df, window):
        df['rp_diff'] = df['last_real_price'] - df['start_price']
        df['sp_diff'] = df['last_sim_price_mean'] - df['start_price']
        df['rp_sp_diff'] = abs(df['sp_diff'] - df['rp_diff'])
        df['above_lb'] = df['last_sim_price_lb'] < df['last_real_price']
        df['below_ub'] = df['last_real_price'] < df['last_sim_price_ub']
        df['in_bounds'] = df.apply(lambda row: row['above_lb'] and row['below_ub'], axis=1)
        df = df[abs(df['sp_diff']) < (window / 2)]
        df = df[abs(df['rp_diff']) < (window / 2)]
        df['rp_dir'] = np.sign(df['rp_diff'])
        df['sp_dir'] = np.sign(df['sp_diff'])
        Evaluation.up_down(df['rp_dir'], "Real")
        Evaluation.up_down(df['sp_dir'], "Sim")
        df['correct_dir'] = df['rp_dir'] == df['sp_dir']

        df['tit_for_tat'] = df['rp_dir'].shift(-1)
        df['tit_for_tat_correct'] = df['rp_dir'] == df['tit_for_tat']
        return df

    @staticmethod
    def print_sim_summary(df):
        total = len(df)
        correct_dirs = len(df[df['rp_dir'] == df['sp_dir']])
        num_inbounds = len(df[df['in_bounds'] == True])
        correct_pc = (correct_dirs / total) * 100
        inbound_pc = (num_inbounds / total) * 100
        print(str(correct_dirs) + "/" + str(total) + " (" + str(correct_pc) + "%) direction predictions were correct")
        print(str(num_inbounds) + "/" + str(total) + " (" + str(inbound_pc) + "%) were in simulation bounds")
        binom_coeff = scipy.special.binom(total, correct_dirs)  # (n..k)
        prob = binom_coeff * (0.5 ** correct_dirs) * (0.5 ** (total - correct_dirs))
        print(str(prob * 100) + "% Probability of getting this result from random chance")
        print(str(1 - scipy.stats.binom.cdf(correct_dirs, total, 0.5)) + " p-value")
        print("Corr coeff " + str(np.corrcoef(df['rp_diff'], df['sp_diff'])))

        tit_for_tat_correct = len(df[df['tit_for_tat_correct'] == True])
        tit_for_tat_correct_pc = tit_for_tat_correct / total
        print(str(tit_for_tat_correct) + "/" + str(total) + " (" + str(tit_for_tat_correct_pc)
              + "%) were correct using tit for tat")

    @staticmethod
    def plot_linregress(x1, x2, window, show_regression_line=True, xlabel="", ylabel=""):
        ling = scipy.stats.linregress(x1, x2)
        print("linregress: " + str(ling))
        slope, intercept, r_value, p_value, std_err = ling
        minimum, maximum = Evaluation.get_plot_min_max(window)
        if show_regression_line:
            x = np.linspace(minimum, maximum)
            plt.plot(x, (x * slope) + intercept)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(x1, x2)

        plt.show()

    @staticmethod
    def up_down(s: pd.Series, description: str):
        num_up = len(s.where(lambda x: x == 1).dropna())
        num_down = len(s.where(lambda x: x == -1).dropna())
        print(description + " num up : " + str(num_up))
        print(description + " num down : " + str(num_down))
