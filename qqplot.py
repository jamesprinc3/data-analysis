import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
import pylab
from typing import List


def plot(observed: List[float], theoretical):

    observed = np.random.normal(0, 1, 1000)
    print(observed)

    rvs = st.alpha.rvs(size=10)
    f, l = np.histogram(rvs)
    print(f)

    sm.qqplot(observed, f)
    pylab.show()

plot([10.0], "henlo")
