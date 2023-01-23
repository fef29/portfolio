from Asset import Asset
from Portfolio import Portfolio
import pandas as pd
from typing import List
from IPython.display import display
import yfinance as yf
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 9

TREASURY_BILL_RATE = 0.11  # %, Jan 2021
EURIBOR_12M = 0.033
TRADING_DAYS_PER_YEAR = 250


def yf_retrieve_data(tickers: List[str], start: datetime, end: datetime):
    dataframes = []

    for ticker_name in tickers:
        # ticker = yf.Ticker(ticker_name)
        # history = ticker.history(period='10y', auto_adjust=True)
        history = yf.download(ticker_name, start=start, end=end)['Adj Close']

        # if history.isnull().any(axis=1).iloc[0]:  # if for DataFrame of OHLC data
        if history.isnull().any():
            history = history.iloc[1:]

        assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
        dataframes.append(history)

    return dataframes


if __name__ == '__main__':

    stocks = ['AAPL', 'AMZN', 'GOOG', 'BRK-B', 'JNJ', 'JPM']
    n = len(stocks)
    eq_weights = np.array([1/n for i in range(n)])

    start = datetime.date(2021, 1, 1)
    end = start + relativedelta(years=2)
    daily_dataframes = yf_retrieve_data(stocks, start, end)

    # Backtesting dates
    start_btt = [start]
    end_btt = [start + relativedelta(years=1)]
    for i in range(4):
        start_btt.append(start_btt[i] + relativedelta(months=3))
        end_btt.append(end_btt[i] + relativedelta(months=3))

    # Walk-forward dates
    start_wf = [start + relativedelta(years=1)]
    end_wf = [start + relativedelta(years=1, months=3)]
    for i in range(4):
        start_wf.append(start_wf[i] + relativedelta(months=3))
        end_wf.append(end_wf[i] + relativedelta(months=3))

    # weights from backtesting
    weights_min_var = []
    weights_risk_tol_100 = []
    for d1, d2 in zip(start_btt, end_btt):
        daily_df_split = [aux_df[d1:d2] for aux_df in daily_dataframes]
        assets = tuple([Asset(name, daily_df) for name, daily_df in zip(stocks, daily_df_split)])
        portfolio = Portfolio(assets)
        # weights minimum variance portfolio
        portfolio.optimize_with_risk_tolerance(0)
        weights_min_var.append(portfolio.weights)
        # weights risk tolerance portfolio
        portfolio.optimize_with_risk_tolerance(100)
        weights_risk_tol_100.append(portfolio.weights)


    # Walkforward
    returns_min = []
    returns_risk = []
    returns_equal = []
    for d1, d2, w_m, w_r in zip(start_wf, end_wf, weights_min_var, weights_risk_tol_100):
        # asset returns
        daily_df_split = [Asset.get_log_period_returns(aux_df[d1:d2]) for aux_df in daily_dataframes]

        returns_m = [asset_returns * _w for asset_returns, _w in zip(daily_df_split, w_m)]
        returns_min.append(sum(returns_m))

        returns_r = [asset_returns * _w for asset_returns, _w in zip(daily_df_split, w_r)]
        returns_risk.append(sum(returns_r))

        returns_eq = [asset_returns * _w for asset_returns, _w in zip(daily_df_split, eq_weights)]
        returns_equal.append(sum(returns_eq))

    # cumsum and plot

    # daily_dataframes = yf_retrieve_data(stocks, start_btt, end_btt)
    assets = tuple([Asset(name, daily_df) for name, daily_df in zip(stocks, daily_dataframes)])
    portfolio = Portfolio(assets)

    pd.options.display.float_format = "{:,.5f}".format

    portfolio = Portfolio(assets)

    portfolio.plot()

    portfolio.optimize_with_risk_tolerance(0)
    riskless_weights = portfolio.weights.flatten()

    portfolio.optimize_with_risk_tolerance(20)
    weights_risk_tolerance = portfolio.weights.flatten()

    portfolio.optimize_with_expected_return(0.25)
    weights_return = portfolio.weights.flatten()

    portfolio.optimize_sharpe_ratio()
    weights_sharpe = portfolio.weights.flatten()

    display(
        pd.DataFrame(
            list(
                zip(
                    [asset.name for asset in portfolio.assets],
                    riskless_weights,
                    weights_risk_tolerance,
                    weights_return,
                    weights_sharpe,
                )
            ),
            columns=[
                'asset',
                'optimize_with_risk_tolerance(0)',
                'optimize_with_risk_tolerance(20)',
                'optimize_with_expected_return(0.25)',
                'optimize_sharpe_ratio()',
            ],
        )
    )

    print('Hello')
