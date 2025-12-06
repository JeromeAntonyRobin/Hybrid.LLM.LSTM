import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Setup Data
ticker = "AAPL"
data = yf.download(ticker, period="2y")
data['Target'] = data['Close'].shift(-1) # Target is tomorrow's price
data.dropna(inplace=True)

# Features: Today's Close
X = data[['Close']].values
y = data['Target'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
last_close = X[-1].reshape(1, -1)
prediction = model.predict(last_close)

print(f"Last Actual Price: {last_close[0][0]}")
print(f"Linear Regression Prediction for Tomorrow: {prediction[0]}")
