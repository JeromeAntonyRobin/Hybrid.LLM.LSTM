import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Hardcoded inputs
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Downloading data for {ticker}...")

# 1. Get Data
df = yf.download(ticker, start=start_date, end=end_date)

# 2. Clean Data (just keep Close price)
df = df[['Close']]

# 3. Save
filename = f"{ticker}_data.csv"
df.to_csv(filename)
print(f"Saved to {filename}")
