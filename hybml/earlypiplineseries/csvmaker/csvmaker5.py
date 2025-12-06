import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Get Data
df = yf.download("AAPL", period="2y")
data = df[['Close']].values

# 2. Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Create Sequences (Lookback 60 days)
lookback = 60
X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train
print("Training LSTM... this may take a moment.")
model.fit(X, y, batch_size=32, epochs=5, verbose=1)

# 6. Predict Next Day
last_60 = scaled_data[-lookback:]
last_60 = last_60.reshape(1, lookback, 1)
pred_scaled = model.predict(last_60)
pred_price = scaler.inverse_transform(pred_scaled)

print(f"LSTM Predicted Price: {pred_price[0][0]}")
