import numpy as np
import pandas as pd
from typing import Tuple
from functools import cache

TREASURY_BILL_RATE = 0.11  # %, Jan 2021
EURIBOR_12M = 0.033
TRADING_DAYS_PER_YEAR = 252


# Needed for type hinting
class Asset:
    pass


class Asset:
    def __init__(self, name: str, daily_price_history: pd.DataFrame):
        self.name = name
        self.daily_returns = self.get_log_period_returns(daily_price_history)
        self.expected_daily_return = np.mean(self.daily_returns)

    @property
    def expected_return(self):
        return TRADING_DAYS_PER_YEAR * self.expected_daily_return

    @staticmethod
    def get_log_period_returns(price_history: pd.Series):
        adj_close = price_history.values
        return np.log(adj_close[1:] / adj_close[:-1]).reshape(-1, 1)

    def __repr__(self):
        return f'<Asset name={self.name}, expected return={self.expected_return}>'
