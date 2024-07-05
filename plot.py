import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt

# Define the stock ticker
ticker = 'TCS.BO'

# Fetch historical data
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Calculate technical indicators
data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# Display the data with new columns
print(data.tail())

# Plot the closing price and moving averages
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['SMA_50'], label='50-Day SMA')
plt.plot(data['SMA_200'], label='200-Day SMA')
plt.title(f'{ticker} Stock Price and Moving Averages')
plt.legend()
plt.show()
