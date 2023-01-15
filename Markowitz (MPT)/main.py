import Asset
import Portfolio
import pandas as pd
from typing import List
from IPython.display import display
import yfinance as yf

from matplotlib import rcParams
rcParams['figure.figsize'] = 12, 9

TREASURY_BILL_RATE = 0.11  # %, Jan 2021
EURIBOR_12M = 0.033
TRADING_DAYS_PER_YEAR = 250


def yf_retrieve_data(tickers: List[str]):
    dataframes = []

    for ticker_name in tickers:
        ticker = yf.Ticker(ticker_name)
        history = ticker.history(period='10y')

        if history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
            history = history.iloc[1:]

        assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
        dataframes.append(history)

    return dataframes


if __name__ == '__main__':

    stocks = ['AAPL', 'AMZN', 'GOOG', 'BRK-B', 'JNJ', 'JPM']
    daily_dataframes = yf_retrieve_data(stocks)
    assets = tuple([Asset(name, daily_df) for name, daily_df in zip(stocks, daily_dataframes)])
    portfolio = Portfolio(assets)

    pd.options.display.float_format = "{:,.5f}".format

    portfolio = Portfolio(assets)
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
