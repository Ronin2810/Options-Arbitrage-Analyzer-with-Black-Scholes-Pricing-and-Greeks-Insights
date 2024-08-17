import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as si
import requests

# Black-Scholes formula
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == "put":
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return price

# Fetch the latest 10-Year Treasury rate from FRED
def fetch_risk_free_rate():
    api_key = 'facfdd6e5c8599cdbf0598a7434a1bf4'
    series_id = 'DGS10'
    base_url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'sort_order': 'desc',
        'limit': 1
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if 'observations' in data and data['observations']:
        return float(data['observations'][0]['value']) / 100  # Convert percentage to decimal
    return 0.05  # Default rate if API call fails

# Get options data
def get_options_data(ticker, expiration_date, risk_free_rate):
    stock = yf.Ticker(ticker)
    options_chain = stock.option_chain(expiration_date)
    
    stock_price = stock.history(period='1d')['Close'][0]
    
    hist_data = stock.history(period="1y")['Close']
    returns = np.log(hist_data / hist_data.shift(1))
    volatility = np.std(returns) * np.sqrt(252)
    
    calls = options_chain.calls
    puts = options_chain.puts

    options_data = pd.merge(calls[['strike', 'lastPrice']], puts[['strike', 'lastPrice']], on='strike', suffixes=('_call', '_put'))
    options_data['time_to_maturity'] = (pd.to_datetime(expiration_date) - datetime.today()).days / 365.0
    options_data['stock_price'] = stock_price
    options_data['volatility'] = volatility

    options_data[['LHS (C - P)', 'RHS (S - K * e^(-r(T-t)))', 'Difference', 'Suggested Action',
                  'Theoretical Call', 'Theoretical Put']] = options_data.apply(
        lambda row: check_put_call_parity_and_theoretical_value(
            row['lastPrice_call'], 
            row['lastPrice_put'], 
            row['stock_price'], 
            row['strike'], 
            row['time_to_maturity'], 
            risk_free_rate,
            row['volatility']
        ), axis=1, result_type='expand'
    )

    # Rearrange columns
    options_data = options_data[['strike', 'lastPrice_call', 'Theoretical Call', 'lastPrice_put', 'Theoretical Put', 
                                 'LHS (C - P)', 'RHS (S - K * e^(-r(T-t)))', 'Difference', 'Suggested Action']]
    
    arbitrage_opportunities = options_data[options_data['Difference'] > trading_cost]
    time_to_maturity = options_data['time_to_maturity'].iloc[0]
    return arbitrage_opportunities, stock_price, time_to_maturity, volatility

# Check put-call parity and calculate theoretical values
def check_put_call_parity_and_theoretical_value(call_price, put_price, stock_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    lhs = call_price - put_price
    rhs = stock_price - strike_price * np.exp(-risk_free_rate * time_to_maturity)
    difference = lhs - rhs
    
    if difference > 0.01:
        action = "Sell Call, Buy Put, Buy Stock, Short Bond"
    elif difference < -0.01:
        action = "Buy Call, Sell Put, Short Stock, Buy Bond"
    else:
        action = "No Arbitrage"
    
    theoretical_call = black_scholes_price(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type="call")
    theoretical_put = black_scholes_price(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type="put")
    
    return lhs, rhs, np.abs(difference), action, theoretical_call, theoretical_put

# Streamlit App
st.title("Options Arbitrage Finder with Black-Scholes Pricing")
st.header("Check for Arbitrage Opportunities using Put-Call Parity and Theoretical Option Pricing")

ticker = st.text_input("Enter the Stock Ticker:", "AAPL")
expiration_date = st.date_input("Select Option Expiration Date:", datetime(2024, 9, 20))
trading_cost = st.number_input("Enter Expected Trading Costs:", min_value=0.0, step=0.01, value=0.01)

risk_free_rate = fetch_risk_free_rate()

if st.button("Find Arbitrage Opportunities"):
    try:
        arbitrage_opportunities, stock_price, time_to_maturity, volatility = get_options_data(ticker, expiration_date.strftime('%Y-%m-%d'), risk_free_rate)
        
        if not arbitrage_opportunities.empty:
            st.success("Arbitrage Opportunities Found!")
            st.markdown(f"### Current Stock Price: ${stock_price:.2f}")
            st.markdown(f"### Time to Maturity: {time_to_maturity:.2f} years")
            st.markdown(f"### Volatility: {volatility:.2%} (Annualized)")
            
            st.dataframe(arbitrage_opportunities.style.format({
                'lastPrice_call': "{:.2f}",
                'Theoretical Call': "{:.2f}",
                'lastPrice_put': "{:.2f}",
                'Theoretical Put': "{:.2f}",
                'LHS (C - P)': "{:.2f}",
                'RHS (S - K * e^(-r(T-t)))': "{:.2f}",
                'Difference': "{:.2f}"
            }))
            
            st.subheader("Difference Between LHS and RHS for Each Strike Price")
            plt.figure(figsize=(10, 6))
            plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['LHS (C - P)'], label='LHS (C - P)', marker='o', color='cyan')
            plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['RHS (S - K * e^(-r(T-t)))'], label='RHS (S - K * e^(-r(T-t)))', marker='x', color='yellow')
            plt.xlabel('Strike Price')
            plt.ylabel('Value')
            plt.title('LHS vs. RHS')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            st.subheader("Theoretical Call vs. Actual Call Prices")
            plt.figure(figsize=(10, 6))
            plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['Theoretical Call'], label="Theoretical Call", marker='o', linestyle='-', color='cyan')
            plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['lastPrice_call'], label="Actual Call", marker='x', linestyle='--', color='magenta')
            plt.xlabel('Strike Price')
            plt.ylabel('Price')
            plt.title('Theoretical Call vs. Actual Call Prices')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            
            st.subheader("Theoretical Put vs. Actual Put Prices")
            plt.figure(figsize=(10, 6))
            plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['Theoretical Put'], label="Theoretical Put", marker='o', linestyle='-', color='cyan')
            plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['lastPrice_put'], label="Actual Put", marker='x', linestyle='--', color='magenta')
            plt.xlabel('Strike Price')
            plt.ylabel('Price')
            plt.title('Theoretical Put vs. Actual Put Prices')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            
            st.subheader("Download Data")
            file_name = f"{ticker}_{expiration_date.strftime('%Y-%m-%d')}_arbitrage_opportunities.csv"
            csv = arbitrage_opportunities.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=file_name,
                mime='text/csv'
            )

        else:
            st.warning("No Arbitrage Opportunities Found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
