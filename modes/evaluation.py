import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.special


class Evaluation:

    @staticmethod
    def load(path: str):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def get_plt(window):
        plt.figure(figsize=(12, 8))

        maximum, minimum = Evaluation.get_min_max(window)
        # plt.xlim(minimum, maximum)
        # plt.ylim(minimum, maximum)

        plt.plot([minimum, maximum], [minimum, maximum])

        return plt

    @staticmethod
    def get_min_max(window):
        minimum = -window / 2
        maximum = window / 2
        return maximum, minimum

    @staticmethod
    def correlate(df, window: int = 10):
        plt = Evaluation.get_plt(window)

        df['rp_diff'] = df['last_real_price'] - df['start_price']
        df['sp_diff'] = df['last_sim_price_mean'] - df['start_price']

        df['above_lb'] = df['last_sim_price_lb'] < df['last_real_price']
        df['below_ub'] = df['last_real_price'] < df['last_sim_price_ub']
        df['in_bounds'] = df.apply(lambda row: row['above_lb'] and row['below_ub'], axis=1)

        df = df[abs(df['sp_diff']) < (window / 2)]
        df = df[abs(df['rp_diff']) < (window / 2)]

        df['rp_dir'] = np.sign(df['rp_diff'])
        df['sp_dir'] = np.sign(df['sp_diff'])

        print(df)

        Evaluation.up_down(df['rp_dir'], "Real")
        Evaluation.up_down(df['sp_dir'], "Sim")

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

        ling = scipy.stats.linregress(df['rp_diff'], df['sp_diff'])
        print("linregress: " + str(ling))
        slope, intercept, r_value, p_value, std_err = ling

        minimum, maximum = Evaluation.get_min_max(window)
        x = np.linspace(minimum, maximum)

        plt.plot(x, (x * slope) + intercept)

        plt.scatter(df['rp_diff'], df['sp_diff'])

        plt.show()

    @staticmethod
    def up_down(s: pd.Series, description: str):
        num_up = len(s.where(lambda x: x == 1).dropna())
        num_down = len(s.where(lambda x: x == -1).dropna())
        print(description + " num up : " + str(num_up))
        print(description + " num down : " + str(num_down))
