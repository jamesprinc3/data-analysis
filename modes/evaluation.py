import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class Evaluation:

    @staticmethod
    def load(path: str):
        df = pd.read_csv(path)
        return df

    @staticmethod
    def correlate(df, window: int = 400):
        plt.figure(figsize=(12, 8))

        minimum = -window / 2
        maximum = window / 2
        # plt.xlim(minimum, maximum)
        # plt.ylim(minimum, maximum)

        plt.plot([minimum, maximum], [minimum, maximum])

        df['rp_diff'] = df['last_real_price'] - df['start_price']
        df['sp_diff'] = df['last_sim_price'] - df['start_price']

        df = df[abs(df['sp_diff']) < (window / 2)]
        df = df[abs(df['rp_diff']) < (window / 2)]

        df['rp_dir'] = np.sign(df['rp_diff'])
        df['sp_dir'] = np.sign(df['sp_diff'])

        print(df)

        total_dirs = len(df)
        correct_dirs = len(df[df['rp_dir'] == df['sp_dir']])

        print(str(correct_dirs) + "/" + str(total_dirs) + " direction predictions were correct")

        print("Corr coeff " + str(np.corrcoef(df['rp_diff'], df['sp_diff'])))

        ling = scipy.stats.linregress(df['rp_diff'], df['sp_diff'])
        print("linregress: " + str(ling))
        slope, intercept, r_value, p_value, std_err = ling

        x = np.linspace(minimum, maximum)

        plt.plot(x, (x*slope) + intercept)

        plt.scatter(df['rp_diff'], df['sp_diff'])

        plt.show()

