import numpy as np
import pandas as pd
from typing import List, Tuple
from functools import cache

TREASURY_BILL_RATE = 0.11  # %, Jan 2021
EURIBOR_12M = 0.033
TRADING_DAYS_PER_YEAR = 250


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

    def __repr__(self):
        return f'<Asset name={self.name}, expected return={self.expected_return}>'

    @staticmethod
    def get_log_period_returns(price_history: pd.DataFrame):
        close = price_history['Close'].values
        return np.log(close[1:] / close[:-1]).reshape(-1, 1)

    @staticmethod
    @cache
    def covariance_matrix(assets: Tuple[Asset]):  # tuple for hashing in the cache
        product_expectation = np.zeros((len(assets), len(assets)))
        for i in range(len(assets)):
            for j in range(len(assets)):
                if i == j:
                    product_expectation[i][j] = np.mean(assets[i].daily_returns * assets[j].daily_returns)
                else:
                    product_expectation[i][j] = np.mean(assets[i].daily_returns @ assets[j].daily_returns.T)

        product_expectation *= (TRADING_DAYS_PER_YEAR - 1) ** 2

        expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
        product_of_expectations = expected_returns @ expected_returns.T

        return product_expectation - product_of_expectations