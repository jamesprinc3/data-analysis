import pandas as pd

class DataUtils:

    @staticmethod
    def date_to_unix(s, unit: str):
        return pd.to_datetime(s, unit=unit).value

    @staticmethod
    def keep_n_std_dev(data: pd.Series, n: int) -> pd.Series:
        return data[~((data - data.mean()).abs() > n * data.std())]