import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class StockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LSTM Stock Predictor (Alpha)")
        
        tk.Label(root, text="Ticker:").pack()
        self.entry = tk.Entry(root)
        self.entry.pack()
        self.entry.insert(0, "AAPL")
        
        tk.Button(root, text="Run Prediction", command=self.run_prediction).pack(pady=10)
        self.result_label = tk.Label(root, text="Result: ...")
        self.result_label.pack(pady=10)

    def run_prediction(self):
        ticker = self.entry.get()
        try:
            # Quick Data Fetch
            df = yf.download(ticker, period="1y")
            if df.empty: raise ValueError("No Data")
            
            data = df[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Simple Sequence Creation
            lookback = 60
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
                
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Model
            model = Sequential()
            model.add(LSTM(50, input_shape=(X.shape[1], 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            
            # Predict
            last_seq = scaled_data[-lookback:].reshape(1, lookback, 1)
            pred = scaler.inverse_transform(model.predict(last_seq))
            
            self.result_label.config(text=f"Predicted: ${pred[0][0]:.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = StockApp(root)
    root.mainloop()
