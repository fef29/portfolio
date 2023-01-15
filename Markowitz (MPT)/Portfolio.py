import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple
import matplotlib.pyplot as plt
import Asset

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 9

TREASURY_BILL_RATE = 0.11  # %, Jan 2021
EURIBOR_12M = 0.033
TRADING_DAYS_PER_YEAR = 250


class Portfolio:
    def __init__(self, assets: Tuple[Asset]):
        self.assets = assets
        self.asset_expected_returns = np.array([asset.expected_return for asset in assets]).reshape(-1, 1)
        self.covariance_matrix = Asset.covariance_matrix(assets)
        self.weights = self.random_weights(len(assets))

    @staticmethod
    def random_weights(weight_count):
        weights = np.random.random((weight_count, 1))
        weights /= np.sum(weights)
        return weights.reshape(-1, 1)

    def unsafe_optimize_with_risk_tolerance(self, risk_tolerance: float):
        res = minimize(
            lambda w: self._variance(w) - risk_tolerance * self._expected_return(w),
            self.random_weights(self.weights.size),
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_with_risk_tolerance(self, risk_tolerance: float):
        assert risk_tolerance >= 0.
        return self.unsafe_optimize_with_risk_tolerance(risk_tolerance)

    def optimize_with_expected_return(self, expected_portfolio_return: float):
        res = minimize(
            lambda w: self._variance(w),
            self.random_weights(self.weights.size),
            method='SLSQP',
            bounds=[(0., 1.) for i in range(self.weights.size)],
            constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
                         {'type': 'eq', 'fun': lambda w: self._expected_return(w) - expected_portfolio_return}]
            )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def optimize_sharpe_ratio(self):
        # Maximize Sharpe ratio = minimize minus Sharpe ratio
        res = minimize(
            lambda w: -(self._expected_return(w) - TREASURY_BILL_RATE / 100) / np.sqrt(self._variance(w)),
            self.random_weights(self.weights.size),
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            bounds=[(0., 1.) for i in range(self.weights.size)]
        )

        assert res.success, f'Optimization failed: {res.message}'
        self.weights = res.x.reshape(-1, 1)

    def _expected_return(self, w):
        return (self.asset_expected_returns.T @ w.reshape(-1, 1))[0][0]

    def _variance(self, w):
        return (w.reshape(-1, 1).T @ self.covariance_matrix @ w.reshape(-1, 1))[0][0]

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
        for i in range(3000):
            portfolio = Portfolio(self.assets)
            X.append(np.sqrt(portfolio.variance))
            y.append(portfolio.expected_return)

        plt.scatter(X, y, label='Random portfolios')

        # Drawing the efficient frontier
        X = []
        y = []
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


