import numpy as np
from scipy.optimize import minimize
from typing import Tuple
import matplotlib.pyplot as plt
from Asset import Asset

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 9

TREASURY_BILL_RATE = 0.11  # %, Jan 2021
EURIBOR_12M = 0.033
TRADING_DAYS_PER_YEAR = 250


class Portfolio:
    def __init__(self, assets: Tuple[Asset]):
        self.assets = assets
        self.n = len(assets)
        self.asset_expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
        self.covariance_matrix = self.vcov_matrix()
        self.weights = self.random_weights(self.n)

    @staticmethod
    def random_weights(n_weight: int):
        weights = np.random.random((n_weight, 1))
        weights /= np.sum(weights)
        return weights.reshape(-1, 1)

    def unsafe_optimize_with_risk_tolerance(self, risk_tolerance: float):
        res = minimize(lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
                       self.random_weights(self.weights.size),
                       constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                       bounds=[(0., 1.) for i in range(self.weights.size)])

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_with_risk_tolerance(self, risk_tolerance: float):
        assert risk_tolerance >= 0.
        return self.unsafe_optimize_with_risk_tolerance(risk_tolerance)

    def optimize_with_expected_return(self, expected_portfolio_return: float):
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return})
        bnds = ((0., 1.) for i in range(self.weights.size))
        res = minimize(lambda w: self._variance(w), self.random_weights(self.weights.size), method='SLSQP',
                       bounds=bnds, constraints=cons)

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_sharpe_ratio(self):
        # Maximize Sharpe ratio = minimize minus Sharpe ratio
        res = minimize(lambda w: -(self._expected_return(w) - TREASURY_BILL_RATE / 100) / np.sqrt(self._variance(w)),
                       self.random_weights(self.weights.size),
                       constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                       bounds=[(0., 1.) for i in range(self.weights.size)])

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def _expected_return(self, w):
        return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]

    def _variance(self, w):
        return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

    def vcov_matrix(self):  # tuple for hashing in the cache
        product_expectation = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    product_expectation[i][j] = np.mean(self.assets[i].daily_returns * self.assets[j].daily_returns)
                else:
                    product_expectation[i][j] = np.mean(self.assets[i].daily_returns @ self.assets[j].daily_returns.T)

        product_expectation *= (TRADING_DAYS_PER_YEAR - 1) ** 2

        expected_returns = np.array([asset.expected_return for asset in self.assets]).reshape(-1, 1)
        product_of_expectations = expected_returns @ expected_returns.T

        return product_expectation - product_of_expectations

    @property
    def expected_return(self):
        return self._expected_return(self.weights)

    @property
    def variance(self):
        return self._variance(self.weights)

    def plot(self):
        X = []
        y = []

        # Drawing random portfolios
        for i in range(5000):
            self.weights = Portfolio.random_weights(self.n)
            X.append(np.sqrt(self.variance))
            y.append(self.expected_return)

        plt.scatter(X, y, label='Random portfolios')

        # Drawing the efficient frontier
        X = []
        y = []
        portfolio = Portfolio(self.assets)
        for rt in np.linspace(-300, 200, 1000):
            portfolio.unsafe_optimize_with_risk_tolerance(rt)
            X.append(np.sqrt(portfolio.variance))
            y.append(portfolio.expected_return)

        plt.plot(X, y, 'k', linewidth=3, label='Efficient frontier')

        # Drawing optimized portfolios
        portfolio.optimize_with_risk_tolerance(0)
        plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'm+', markeredgewidth=5, markersize=20,
                 label='optimize_with_risk_tolerance(0)')

        portfolio.optimize_with_risk_tolerance(100)
        plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'r+', markeredgewidth=5, markersize=20,
                 label='optimize_with_risk_tolerance(100)')

        # portfolio.optimize_with_expected_return(0.25)
        # plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'g+', markeredgewidth=5, markersize=20,
        #          label='optimize_with_expected_return(0.25)')

        portfolio.optimize_sharpe_ratio()
        plt.plot(np.sqrt(portfolio.variance), portfolio.expected_return, 'y+', markeredgewidth=5, markersize=20,
                 label='optimize_sharpe_ratio()')

        plt.xlabel('Portfolio standard deviation')
        plt.ylabel('Portfolio expected (logarithmic) return')
        plt.legend(loc='lower right')
        plt.show()

    def __repr__(self):
        return f'<Portfolio assets={[asset.name for asset in self.assets]}, expected return={self.expected_return},' \
               f' variance={self.variance}>'


