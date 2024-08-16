import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as si

# Black-Scholes formula
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == "put":
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return price

# Cache options data retrieval
def get_options_data(ticker, expiration_date, risk_free_rate):
    stock = yf.Ticker(ticker)
    options_chain = stock.option_chain(expiration_date)
    
    # Current stock price
    stock_price = stock.history(period='1d')['Close'][0]
    
    # Fetch historical volatility (annualized standard deviation)
    hist_data = stock.history(period="1y")['Close']
    returns = np.log(hist_data / hist_data.shift(1))
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    # Call and Put options data
    calls = options_chain.calls
    puts = options_chain.puts

    # Combine calls and puts on strike prices
    options_data = pd.merge(calls[['strike', 'lastPrice']], puts[['strike', 'lastPrice']], on='strike', suffixes=('_call', '_put'))
    options_data['time_to_maturity'] = (pd.to_datetime(expiration_date) - datetime.today()).days / 365.0
    options_data['stock_price'] = stock_price  # Add stock price for calculations
    options_data['volatility'] = volatility

    # Calculate theoretical prices and check for arbitrage opportunities
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

    # Filter for arbitrage opportunities
    arbitrage_opportunities = options_data[options_data['Difference'] > 0.01]
    time_to_maturity = options_data['time_to_maturity'].iloc[0]  # Use the first row's time to maturity for display
    return arbitrage_opportunities, stock_price, time_to_maturity, volatility, options_data

# Function to calculate put-call parity, theoretical values, and suggest action
def check_put_call_parity_and_theoretical_value(call_price, put_price, stock_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    # Calculate LHS and RHS
    lhs = call_price - put_price  # Left-Hand Side (LHS)
    rhs = stock_price - strike_price * np.exp(-risk_free_rate * time_to_maturity)  # Right-Hand Side (RHS)
    difference = lhs - rhs
    
    # Determine arbitrage action
    if difference > 0.01:
        action = "Sell Call, Buy Put, Buy Stock, Short Bond"
    elif difference < -0.01:
        action = "Buy Call, Sell Put, Short Stock, Buy Bond"
    else:
        action = "No Arbitrage"
    
    # Calculate theoretical prices using Black-Scholes
    theoretical_call = black_scholes_price(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type="call")
    theoretical_put = black_scholes_price(stock_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type="put")
    
    return lhs, rhs, np.abs(difference), action, theoretical_call, theoretical_put

# Streamlit App

# Title and Header
st.title("Options Arbitrage Finder with Black-Scholes Pricing")
st.header("Check for Arbitrage Opportunities using Put-Call Parity and Theoretical Option Pricing")

# User Inputs with sliders
ticker = st.text_input("Enter the Stock Ticker:", "AAPL")
expiration_date = st.date_input("Select Option Expiration Date:", datetime(2024, 9, 20))
risk_free_rate = st.slider("Enter the Risk-Free Rate (as a decimal):", 0.0, 0.1, 0.05, 0.01)
arbitrage_threshold = st.slider("Enter the Arbitrage Threshold (in $):", 0.0, 1.0, 0.01, 0.01)

if st.button("Find Arbitrage Opportunities"):
    try:
        # Fetch and display arbitrage opportunities
        arbitrage_opportunities, stock_price, time_to_maturity, volatility, options_data = get_options_data(ticker, expiration_date.strftime('%Y-%m-%d'), risk_free_rate)
        
        if not arbitrage_opportunities.empty:
            st.success("Arbitrage Opportunities Found!")
            st.markdown(f"### Current Stock Price: ${stock_price:.2f}")
            st.markdown(f"### Time to Maturity: {time_to_maturity:.2f} years")
            st.markdown(f"### Volatility: {volatility:.2%} (Annualized)")

            # Drop redundant columns
            arbitrage_opportunities = arbitrage_opportunities.drop(columns=['stock_price', 'time_to_maturity', 'volatility'])
            
            st.dataframe(arbitrage_opportunities.style.format({
                'LHS (C - P)': "{:.2f}",
                'RHS (S - K * e^(-r(T-t)))': "{:.2f}",
                'Difference': "{:.2f}",
                'Theoretical Call': "{:.2f}",
                'Theoretical Put': "{:.2f}"
            }))
            
            # Visualization: Difference Plot with dark background and bar borders
            st.subheader("Difference Between LHS and RHS for Each Strike Price")
            with plt.style.context('dark_background'):
                plt.figure(figsize=(10, 6))
                plt.bar(arbitrage_opportunities['strike'], arbitrage_opportunities['Difference'], 
                        color='blue', width=5, edgecolor='white', linewidth=1.5)
                plt.xlabel('Strike Price')
                plt.ylabel('Difference (|LHS - RHS|)')
                plt.title('Arbitrage Opportunities by Strike Price')
                st.pyplot(plt)

            # Visualization: Suggested Actions Plot with dark background
            st.subheader("Suggested Actions Distribution")
            with plt.style.context('dark_background'):
                action_counts = arbitrage_opportunities['Suggested Action'].value_counts()
                plt.figure(figsize=(6, 6))
                wedges, texts, autotexts = plt.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%', 
                                                   colors=['#66b3ff', '#99ff99', '#ff9999', '#ffcc99'], 
                                                   wedgeprops=dict(edgecolor='white'))
                plt.title('Suggested Arbitrage Actions')
                st.pyplot(plt)
            
            # Visualization: Theoretical Call vs. Actual Call Prices
            st.subheader("Theoretical vs. Actual Call Prices")
            with plt.style.context('dark_background'):
                plt.figure(figsize=(10, 6))
                plt.plot(options_data['strike'], options_data['Theoretical Call'], label='Theoretical Call Price', marker='o', color='green')
                plt.plot(options_data['strike'], options_data['lastPrice_call'], label='Actual Call Price', marker='x', color='blue')
                plt.xlabel('Strike Price')
                plt.ylabel('Price')
                plt.title('Theoretical vs. Actual Call Prices')
                plt.legend()
                st.pyplot(plt)
            
            # Visualization: Theoretical Put vs. Actual Put Prices
            st.subheader("Theoretical vs. Actual Put Prices")
            with plt.style.context('dark_background'):
                plt.figure(figsize=(10, 6))
                plt.plot(options_data['strike'], options_data['Theoretical Put'], label='Theoretical Put Price', marker='o', color='red')
                plt.plot(options_data['strike'], options_data['lastPrice_put'], label='Actual Put Price', marker='x', color='purple')
                plt.xlabel('Strike Price')
                plt.ylabel('Price')
                plt.title('Theoretical vs. Actual Put Prices')
                plt.legend()
                st.pyplot(plt)
            
            # Visualization: LHS vs. RHS Comparison with dark background
            st.subheader("LHS vs. RHS Comparison")
            with plt.style.context('dark_background'):
                plt.figure(figsize=(10, 6))
                plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['LHS (C - P)'], 
                         label='LHS (C - P)', marker='o', color='cyan')
                plt.plot(arbitrage_opportunities['strike'], arbitrage_opportunities['RHS (S - K * e^(-r(T-t)))'], 
                         label='RHS (S - K * e^(-r(T-t)))', marker='x', color='yellow')
                plt.xlabel('Strike Price')
                plt.ylabel('Value')
                plt.title('LHS vs. RHS Comparison')
                plt.legend()
                st.pyplot(plt)
            
            # Export Data
            st.subheader("Download Data")
            filename = f"{ticker}_{expiration_date.strftime('%Y%m%d')}_arbitrage_opportunities.csv"
            csv = arbitrage_opportunities.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=filename,
                mime='text/csv'
            )

        else:
            st.warning("No Arbitrage Opportunities Found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
