import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import threading # To keep GUI responsive during training

# --- Data Acquisition (Modified for Web App) ---
# Always download for simplicity in a web app context
def get_stock_data_live(ticker, start_date, end_date):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data downloaded from Yahoo Finance.")
        print(f"Successfully downloaded data for {ticker}.")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Fallback to synthetic data for demonstration if download fails
        print("Generating synthetic data for demonstration purposes.")
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        synthetic_data = pd.DataFrame(index=date_range)
        synthetic_data['Close'] = np.sin(np.linspace(0, 100, len(date_range))) * 1000 + 4000
        synthetic_data['Open'] = synthetic_data['Close'] * 0.99
        synthetic_data['High'] = synthetic_data['Close'] * 1.01
        synthetic_data['Low'] = synthetic_data['Close'] * 0.98
        synthetic_data['Volume'] = np.random.randint(1000000, 5000000, len(date_range))
        return synthetic_data

# --- Data Preprocessing ---
def create_sequences(data, look_back):
    X, Y = [], []
    # Ensure data has enough points for at least one sequence
    if len(data) <= look_back:
        return np.array([]), np.array([])
        
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# --- Model Definitions (Only one model for simplicity in GUI demo) ---
def build_conv1d_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class StockPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("LSTM Stock Price Predictor")
        master.geometry("1200x800") # Adjust window size
        master.resizable(True, True)

        # Configure grid for responsiveness
        master.grid_rowconfigure(0, weight=0) # For controls
        master.grid_rowconfigure(1, weight=1) # For plot
        master.grid_columnconfigure(0, weight=1)

        # --- Controls Frame ---
        self.controls_frame = ttk.LabelFrame(master, text="Configuration")
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.controls_frame.grid_columnconfigure(1, weight=1) # Make entry widgets expand

        # Ticker
        ttk.Label(self.controls_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.ticker_var = tk.StringVar(value="^GSPC")
        ttk.Entry(self.controls_frame, textvariable=self.ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Start Date
        ttk.Label(self.controls_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        today = datetime.date.today()
        ten_years_ago = (today - datetime.timedelta(days=365*10)).strftime('%Y-%m-%d')
        self.start_date_var = tk.StringVar(value=ten_years_ago)
        ttk.Entry(self.controls_frame, textvariable=self.start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        # End Date
        ttk.Label(self.controls_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        yesterday = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        self.end_date_var = tk.StringVar(value=yesterday)
        ttk.Entry(self.controls_frame, textvariable=self.end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Look Back Window
        ttk.Label(self.controls_frame, text="Look Back Window (days):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.controls_frame, textvariable=self.look_back_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        # Train/Test Split Ratio (Slider)
        ttk.Label(self.controls_frame, text="Train/Test Ratio:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.train_ratio_var = tk.DoubleVar(value=0.9)
        self.train_ratio_slider = ttk.Scale(self.controls_frame, from_=0.1, to=0.99, orient="horizontal",
                                            variable=self.train_ratio_var, command=self._update_train_ratio_label)
        self.train_ratio_slider.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        self.train_ratio_label = ttk.Label(self.controls_frame, text=f"{self.train_ratio_var.get():.2f}")
        self.train_ratio_label.grid(row=4, column=2, padx=5, pady=2, sticky="w")

        # Run Prediction Button
        self.run_button = ttk.Button(self.controls_frame, text="Run Prediction", command=self._run_prediction_thread)
        self.run_button.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        # Status Label
        self.status_label = ttk.Label(self.controls_frame, text="Ready.")
        self.status_label.grid(row=6, column=0, columnspan=3, padx=5, pady=2, sticky="w")

        # --- Metrics Frame ---
        self.metrics_frame = ttk.LabelFrame(self.controls_frame, text="Performance Metrics (Conv1D + LSTM)")
        self.metrics_frame.grid(row=0, column=3, rowspan=7, padx=10, pady=10, sticky="nsew")
        self.metrics_frame.grid_columnconfigure(1, weight=1)

        self.metric_labels = {}
        metrics_order = ["MSE", "R2", "MAPE", "Accuracy (0-1)"]
        for i, metric_name in enumerate(metrics_order):
            ttk.Label(self.metrics_frame, text=f"{metric_name}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.metric_labels[metric_name] = ttk.Label(self.metrics_frame, text="N/A")
            self.metric_labels[metric_name].grid(row=i, column=1, padx=5, pady=2, sticky="ew")
        
        ttk.Label(self.metrics_frame, text="Pred. Period:").grid(row=len(metrics_order), column=0, padx=5, pady=2, sticky="w")
        self.metric_labels["Prediction Period"] = ttk.Label(self.metrics_frame, text="N/A")
        self.metric_labels["Prediction Period"].grid(row=len(metrics_order), column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.metrics_frame, text="Num Predictions:").grid(row=len(metrics_order)+1, column=0, padx=5, pady=2, sticky="w")
        self.metric_labels["Num Predictions"] = ttk.Label(self.metrics_frame, text="N/A")
        self.metric_labels["Num Predictions"].grid(row=len(metrics_order)+1, column=1, padx=5, pady=2, sticky="ew")


        # --- Plotting Frame ---
        self.plot_frame = ttk.LabelFrame(master, text="Prediction Plot")
        self.plot_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Initial plot to show something
        self._clear_plot()


    def _update_train_ratio_label(self, val):
        self.train_ratio_label.config(text=f"{float(val):.2f}")

    def _clear_plot(self):
        self.ax.clear()
        self.ax.set_title('Prediction Plot')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Stock Price')
        self.ax.grid(True)
        self.canvas.draw()

    def _run_prediction_thread(self):
        # Disable button to prevent multiple clicks
        self.run_button.config(state=tk.DISABLED)
        self.status_label.config(text="Processing... Please wait.")
        self.master.update_idletasks() # Force GUI update

        # Run the heavy computation in a separate thread
        thread = threading.Thread(target=self._perform_prediction)
        thread.start()

    def _perform_prediction(self):
        try:
            # Get values from GUI inputs
            ticker = self.ticker_var.get()
            start_date_str = self.start_date_var.get()
            end_date_str = self.end_date_var.get()
            look_back = self.look_back_var.get()
            train_test_split_ratio = self.train_ratio_var.get()

            # Input validation
            if not ticker or not start_date_str or not end_date_str:
                raise ValueError("All fields are required.")
            
            try:
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                if start_date >= end_date:
                    raise ValueError("Start date must be before end date.")
            except ValueError:
                raise ValueError("Invalid date format. Use YYYY-MM-DD.")
            
            if look_back <= 0:
                raise ValueError("Look back window must be a positive integer.")
            if not (0.1 <= train_test_split_ratio <= 0.99):
                raise ValueError("Train/Test ratio must be between 0.1 and 0.99.")

            # --- Data Acquisition ---
            self.status_label.config(text="Downloading data...")
            self.master.update_idletasks()
            data = get_stock_data_live(ticker, start_date, end_date)
            data = data.sort_index()

            features = ['Close']
            data_for_scaling = data[features].values

            # --- Data Preprocessing ---
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_for_scaling)

            X, Y = create_sequences(scaled_data, look_back)
            
            if len(X) == 0:
                raise ValueError("Not enough data to create sequences with the given look-back period. Try reducing look-back or extending data range.")

            train_size = int(len(X) * train_test_split_ratio)
            
            if train_size < 1 or (len(X) - train_size) < 1: # Ensure at least 1 sample in train and test
                raise ValueError("Invalid train/test split. Ensure at least one sample for training and one for testing. Adjust ratio or data range.")

            X_train, X_test = X[0:train_size], X[train_size:len(X)]
            Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

            # Reshape input to be [samples, time_steps, features] for LSTM
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            test_dates = data.index[train_size + look_back:]
            if len(test_dates) == 0:
                raise ValueError("No data available for the test (prediction) period. Adjust date range or train/test ratio.")


            # --- Model Training ---
            self.status_label.config(text="Training model (Conv1D + LSTM)...")
            self.master.update_idletasks()
            model = build_conv1d_lstm_model((look_back, 1))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stopping])

            # --- Make Predictions ---
            self.status_label.config(text="Making predictions...")
            self.master.update_idletasks()
            test_predict_scaled = model.predict(X_test)

            # Inverse transform predictions and actual values to original scale
            dummy_predictions = np.zeros((len(test_predict_scaled), len(features)))
            dummy_predictions[:, 0] = test_predict_scaled.flatten()
            test_predict = scaler.inverse_transform(dummy_predictions)[:, 0]

            dummy_Y_test = np.zeros((len(Y_test), len(features)))
            dummy_Y_test[:, 0] = Y_test.flatten()
            actual_values = scaler.inverse_transform(dummy_Y_test)[:, 0]

            # --- Evaluate Metrics ---
            mse = mean_squared_error(actual_values, test_predict)
            mape = np.mean(np.abs((actual_values - test_predict) / np.where(actual_values == 0, 1e-9, actual_values))) * 100
            r2 = r2_score(actual_values, test_predict)
            accuracy_score_0_1 = max(0, 1 - (mape / 100))

            # --- Update GUI Metrics ---
            self.metric_labels["MSE"].config(text=f"{mse:.4f}")
            self.metric_labels["R2"].config(text=f"{r2:.4f}")
            self.metric_labels["MAPE"].config(text=f"{mape:.2f}%")
            self.metric_labels["Accuracy (0-1)"].config(text=f"{accuracy_score_0_1:.4f}")
            self.metric_labels["Prediction Period"].config(text=f"{test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
            self.metric_labels["Num Predictions"].config(text=f"{len(test_dates)}")

            # --- Generate and Update Plot ---
            self.ax.clear()
            self.ax.plot(test_dates, actual_values, label='Actual Values', color='black', linewidth=2)
            self.ax.plot(test_dates, test_predict, label='Conv1D + LSTM Predicted', color='purple', linestyle='--')
            self.ax.set_title(f'{ticker} Stock Price Prediction: Actual vs. Predicted Values')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Stock Price')
            self.ax.legend()
            self.ax.grid(True)
            self.fig.tight_layout()
            self.canvas.draw()
            
            self.status_label.config(text="Prediction complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Error occurred.")
            self._clear_plot() # Clear plot on error
        finally:
            self.run_button.config(state=tk.NORMAL) # Re-enable button


def main():
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()