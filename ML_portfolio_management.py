import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import talib

# Define the time range
start_date = '2000-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

equity_etfs = ['SPY', 'VEIEX', 'VYM']
high_yield_bond_etf = 'VWEHX'
high_dividend_yield_etf = 'VHDYX'
debt_etfs = ['VBMFX', 'VUSTX', 'PREMX']
gold_etf = 'GLD'
energy_etf = 'GSG'
sp500_symbol = '^GSPC'

# Download the adjusted close prices
asset_prices = yf.download(equity_etfs + debt_etfs + [gold_etf, energy_etf, sp500_symbol], start=start_date, end=end_date)['Adj Close']
asset_prices = asset_prices.dropna()

using_machine_learning = False

# Initialize weights DataFrame
weights = pd.DataFrame(columns=asset_prices.columns) #index=asset_prices.index, 
asset_returns = pd.DataFrame(columns=asset_prices.columns) #index=asset_prices.index, 

init_idx = 180

for i in range(init_idx, len(asset_prices)-30):

    if asset_prices.index[i].month != asset_prices.index[i-1].month:
        print(f"Conductiong test for {dates[i]}")
        if len(asset_returns)>0:
            print((asset_returns+1).cumprod().iloc[-1])
    for asset in asset_prices.columns:
        # Iterate over each period and perform walk-forward validation
        close_prices = asset_prices[asset]
        dates = close_prices.index

        if dates[i].month != dates[i-1].month:

            long_term_ema = close_prices.iloc[i-160:i+1].ewm(span=150).mean()
            price = close_prices.iloc[i]
            if price < long_term_ema.iloc[-1]:
                if using_machine_learning == False:
                    weights.loc[close_prices.index[i], asset] = 0
            else:
                weights.loc[close_prices.index[i], asset] = 1

            # if we have last month's signal (weight[asset].iloc[-2])
            if len(weights)>1 and close_prices.index[i]>dates[init_idx]: 
                if weights[asset].iloc[-2] == 1: # if last month's signal is 1, meaning staying in the market
                    # get the return 
                    try:
                        asset_returns.loc[close_prices.index[i], asset] = (close_prices[i]-close_prices[i-1])/close_prices[i-1]
                    except:
                        asset_returns.loc[close_prices.index[i], asset] = 0
                else:
                    asset_returns.loc[close_prices.index[i], asset] = 0

        else:
            if len(weights)>1 and close_prices.index[i]>dates[init_idx]: 
                weights.loc[close_prices.index[i], asset] = weights.loc[close_prices.index[i-1], asset]
                try:
                    asset_returns.loc[close_prices.index[i], asset] = (close_prices[i]-close_prices[i-1])/close_prices[i-1]
                    
                except:
                    asset_returns.loc[close_prices.index[i], asset] = 0
                    
                if weights[asset].iloc[-2] == 1:
                    # get the return 
                    asset_returns.loc[close_prices.index[i], asset] = (close_prices[i]-close_prices[i-1])/close_prices[i-1]
                else:
                    asset_returns.loc[close_prices.index[i], asset] = 0

print("Calculating the return now")
#Calculate portfolio returns
equity_weights = 0.6 / len(equity_etfs)
bond_weights = 0.4 / len(debt_etfs)
portfolio_returns = (asset_returns[equity_etfs] * equity_weights).sum(axis=1) + (asset_returns[debt_etfs] * bond_weights).sum(axis=1)
spy_prices = asset_prices['SPY']

# Calculate portfolio cumulative returns based on raw prices
portfolio_cum_returns = (portfolio_returns+1).cumprod()

# Calculate SPY cumulative returns based on raw prices
spy_cum_returns = (spy_prices / spy_prices.iloc[0])

# Calculate metrics for portfolio
# Calculate portfolio annual return based on cumulative returns
portfolio_annual_return = portfolio_cum_returns.iloc[-1] ** (252 / len(portfolio_cum_returns)) - 1

# Calculate portfolio annual volatility based on returns
portfolio_annual_volatility = portfolio_returns.std() * np.sqrt(252)

# Calculate portfolio maximum drawdown based on cumulative returns
portfolio_max_drawdown = (portfolio_cum_returns / portfolio_cum_returns.cummax() - 1).min()

# Calculate the percentage of profitable months for the portfolio based on monthly returns
portfolio_monthly_returns = portfolio_returns.resample('M').sum()
portfolio_profitable_months = (portfolio_monthly_returns > 0).mean() * 100

portfolio_annual_returns = portfolio_returns.resample('Y').sum()
portfolio_profitable_years = (portfolio_annual_returns > 0).mean() * 100

portfolio_sharpe_ratio = portfolio_annual_return / portfolio_annual_volatility

# Calculate SPY annual return based on prices
spy_annual_return = (spy_prices[-1] / spy_prices[0]) ** (252 / len(spy_prices)) - 1
spy_annual_volatility = spy_prices.pct_change().std() * np.sqrt(252)
spy_max_drawdown = ((spy_prices / spy_prices.cummax()) - 1).min()
# Calculate the percentage of profitable months for SPY based on prices
# Calculate the number of profitable months for SPY based on prices
spy_monthly_prices = spy_prices.resample('M').last()
spy_profitable_months = (spy_monthly_prices.pct_change() > 0).sum()
total_months = len(spy_monthly_prices)
spy_num_profitable_months = spy_profitable_months.sum()/total_months* 100
# Calculate the percentage of profitable years for SPY based on prices
spy_prices_yearly = spy_prices.resample('Y').last()
spy_profitable_years = (spy_prices_yearly > spy_prices_yearly.shift(1)).mean() * 100

spy_sharpe_ratio = spy_annual_return / spy_annual_volatility

# Create a summary table
summary_table = pd.DataFrame({
    'Metric': ['Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Profitable Months (%)', 'Profitable Years (%)', 'Sharpe Ratio'],
    'Portfolio': [portfolio_annual_return, portfolio_annual_volatility, portfolio_max_drawdown, portfolio_profitable_months, portfolio_profitable_years, portfolio_sharpe_ratio],
    'SPY': [spy_annual_return, spy_annual_volatility, spy_max_drawdown, spy_num_profitable_months, spy_profitable_years, spy_sharpe_ratio]
})

print(summary_table)

leverage_factor = spy_annual_volatility/portfolio_annual_volatility
# Calculate portfolio cumulative returns based on raw prices
leverage_portfolio_cum_returns = ((portfolio_returns*2)+1).cumprod()

# Plot portfolio and SPY cumulative returns
plt.plot(leverage_portfolio_cum_returns, label=f'{leverage_factor} Leveraged Portfolio')
plt.plot(portfolio_cum_returns, label='Portfolio')
plt.plot(spy_cum_returns, label='SPY')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Portfolio vs SPY Cumulative Returns')
plt.legend()
plt.show()

print()