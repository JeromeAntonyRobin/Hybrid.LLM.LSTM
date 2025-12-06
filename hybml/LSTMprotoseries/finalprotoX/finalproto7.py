import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import threading
import os
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- Data Acquisition ---
def get_stock_data_live(ticker, start_date, end_date):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            # Check if end_date is in the future. yfinance will return empty if end_date is future.
            # Convert string dates to datetime objects for comparison
            start_dt = datetime.datetime.strptime(str(start_date), "%Y-%m-%d").date()
            end_dt = datetime.datetime.strptime(str(end_date), "%Y-%m-%d").date()
            if end_dt > datetime.date.today():
                 print(f"No data available from Yahoo Finance for dates beyond today ({datetime.date.today().strftime('%Y-%m-%d')}). This is expected for future dates.")
                 return pd.DataFrame() # Return empty for future dates, not an error
            else:
                raise ValueError("No data downloaded from Yahoo Finance for historical period.")
        print(f"Successfully downloaded data for {ticker}.")
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Fallback to synthetic data for demonstration if download fails for past dates
        print("Generating synthetic data as a fallback...")
        dates = pd.bdate_range(start=start_date, end=end_date)
        if dates.empty:
            print("No business days in the specified range for synthetic data.")
            return pd.DataFrame()
        np.random.seed(42) # for reproducibility
        close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        data = pd.DataFrame({'Close': close_prices}, index=dates)
        data.index.name = 'Date'
        print("Synthetic data generated.")
        return data


# --- Data Preprocessing ---
def create_sequences(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# --- Model Architectures ---
def build_lstm_model(input_shape, lstm_units=50, num_lstm_layers=1, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    for i in range(num_lstm_layers):
        if i == 0:
            model.add(LSTM(units=lstm_units, return_sequences=(num_lstm_layers > 1), input_shape=input_shape))
        elif i == num_lstm_layers - 1:
            model.add(LSTM(units=lstm_units)) # Last LSTM layer doesn't return sequences
        else:
            model.add(LSTM(units=lstm_units, return_sequences=True))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def build_bidirectional_lstm_model(input_shape, lstm_units=50, num_lstm_layers=1, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    for i in range(num_lstm_layers):
        if i == 0:
            model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=(num_lstm_layers > 1)), input_shape=input_shape))
        elif i == num_lstm_layers - 1:
            model.add(Bidirectional(LSTM(units=lstm_units)))
        else:
            model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def build_cnn_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

model_builders = {
    "LSTM": build_lstm_model,
    "Bidirectional LSTM": build_bidirectional_lstm_model,
    "CNN-LSTM": build_cnn_lstm_model
}

class StockPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Stock Price Predictor")
        master.geometry("1200x800")

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self._create_main_tab()
        self._create_averaging_tab()
        self._create_absolute_testing_tab()
        self._create_experiment_tab() # Added missing experiment tab creation from previous version
        self._create_future_prediction_tab() # NEW: Future Prediction Tab

        self.running_thread = None # To keep track of running prediction thread

    # --- Main Tab ---
    def _create_main_tab(self):
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main Prediction")

        # Left Panel (Controls)
        self.left_panel_main = ttk.LabelFrame(self.main_tab, text="Configuration")
        self.left_panel_main.pack(side="left", fill="y", padx=10, pady=10)

        # Ticker Symbol
        ttk.Label(self.left_panel_main, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.ticker_var = tk.StringVar(value="GOOGL")
        ttk.Entry(self.left_panel_main, textvariable=self.ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Start Date
        ttk.Label(self.left_panel_main, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.start_date_var = tk.StringVar(value="2010-01-01")
        ttk.Entry(self.left_panel_main, textvariable=self.start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        # End Date
        ttk.Label(self.left_panel_main, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.end_date_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(self.left_panel_main, textvariable=self.end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Look-back Period
        ttk.Label(self.left_panel_main, text="Look-back Period:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.left_panel_main, textvariable=self.look_back_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        # Train/Test Split
        ttk.Label(self.left_panel_main, text="Train/Test Split (%):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.train_test_split_var = tk.IntVar(value=80)
        ttk.Entry(self.left_panel_main, textvariable=self.train_test_split_var).grid(row=4, column=1, padx=5, pady=2, sticky="ew")

        # Model Architecture
        ttk.Label(self.left_panel_main, text="Model Architecture:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.model_architecture_var = tk.StringVar(value="LSTM")
        ttk.OptionMenu(self.left_panel_main, self.model_architecture_var, "LSTM", *model_builders.keys()).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # LSTM Units
        ttk.Label(self.left_panel_main, text="LSTM Units:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.lstm_units_var = tk.IntVar(value=50)
        ttk.Entry(self.left_panel_main, textvariable=self.lstm_units_var).grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        # Number of LSTM Layers
        ttk.Label(self.left_panel_main, text="LSTM Layers:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.num_lstm_layers_var = tk.IntVar(value=1)
        ttk.Entry(self.left_panel_main, textvariable=self.num_lstm_layers_var).grid(row=7, column=1, padx=5, pady=2, sticky="ew")

        # Dropout Rate
        ttk.Label(self.left_panel_main, text="Dropout Rate:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.dropout_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.left_panel_main, textvariable=self.dropout_rate_var).grid(row=8, column=1, padx=5, pady=2, sticky="ew")

        # Learning Rate
        ttk.Label(self.left_panel_main, text="Learning Rate:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(self.left_panel_main, textvariable=self.learning_rate_var).grid(row=9, column=1, padx=5, pady=2, sticky="ew")

        # Run Prediction Button
        self.run_button = ttk.Button(self.left_panel_main, text="Run Main Prediction", command=self._start_main_prediction_thread)
        self.run_button.grid(row=10, column=0, columnspan=2, pady=10)

        # Status Label
        self.status_label = ttk.Label(self.left_panel_main, text="Ready.")
        self.status_label.grid(row=11, column=0, columnspan=2, pady=5)

        # Metrics Display
        self.metrics_frame = ttk.LabelFrame(self.left_panel_main, text="Prediction Metrics")
        self.metrics_frame.grid(row=12, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        self.metric_labels = {}
        metrics = ["MSE", "R2", "MAPE", "Accuracy (0-1)", "Prediction Period", "Num Predictions"]
        for i, metric in enumerate(metrics):
            ttk.Label(self.metrics_frame, text=f"{metric}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.metric_labels[metric] = ttk.Label(self.metrics_frame, text="N/A")
            self.metric_labels[metric].grid(row=i, column=1, padx=5, pady=2, sticky="w")


        # Right Panel (Plot)
        self.right_panel_main = ttk.Frame(self.main_tab)
        self.right_panel_main.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig_main, self.ax_main = plt.subplots(figsize=(8, 6))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=self.right_panel_main)
        self.canvas_main_widget = self.canvas_main.get_tk_widget()
        self.canvas_main_widget.pack(side="top", fill="both", expand=True)

        self.toolbar_main = NavigationToolbar2Tk(self.canvas_main, self.toolbar_main_frame)
        self.toolbar_main.update()
        self.canvas_main_widget.pack(side="top", fill="both", expand=True)

    def _start_main_prediction_thread(self):
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)
        self.run_future_button.config(state=tk.DISABLED) # Disable future prediction button too

        self.status_label.config(text="Prediction in progress...")
        self.running_thread = threading.Thread(target=self._run_main_prediction)
        self.running_thread.start()

    def _run_main_prediction(self):
        try:
            ticker = self.ticker_var.get()
            start_date = self.start_date_var.get()
            end_date = self.end_date_var.get()
            look_back = self.look_back_var.get()
            train_test_split_ratio = self.train_test_split_var.get() / 100
            model_architecture = self.model_architecture_var.get()
            lstm_units = self.lstm_units_var.get()
            num_lstm_layers = self.num_lstm_layers_var.get()
            dropout_rate = self.dropout_rate_var.get()
            learning_rate = self.learning_rate_var.get()

            # Data Acquisition
            data = get_stock_data_live(ticker, start_date, end_date)
            if data.empty:
                messagebox.showerror("Error", "No data downloaded for main prediction.")
                return

            scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['Close'].values.reshape(-1, 1))

            X, Y = create_sequences(scaled_data, look_back)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            train_size = int(len(X) * train_test_split_ratio)
            X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
            Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

            # Model Building and Training
            model_builder = model_builders[model_architecture]
            model = model_builder(
                input_shape=(look_back, 1),
                lstm_units=lstm_units,
                num_lstm_layers=num_lstm_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )

            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])

            # Making predictions on the test set
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # Invert predictions to original scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data['Close'].values.reshape(-1, 1)) # Fit scaler on original full data range

            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)
            Y_train_actual = scaler.inverse_transform(Y_train.reshape(-1, 1))
            Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

            # Metric calculation
            # MSE
            train_mse = mean_squared_error(Y_train_actual, train_predict)
            test_mse = mean_squared_error(Y_test_actual, test_predict)
            # R2 Score
            train_r2 = r2_score(Y_train_actual, train_predict)
            test_r2 = r2_score(Y_test_actual, test_predict)
            # MAPE
            train_mape = np.mean(np.abs((Y_train_actual - train_predict) / Y_train_actual)) * 100
            test_mape = np.mean(np.abs((Y_test_actual - test_predict) / Y_test_actual)) * 100
            # Accuracy (1 - decimal MAPE)
            train_accuracy = 1 - (np.abs((Y_train_actual - train_predict) / Y_train_actual)).mean()
            test_accuracy = 1 - (np.abs((Y_test_actual - test_predict) / Y_test_actual)).mean()

            # Shift train predictions for plotting
            train_predict_plot = np.empty_like(scaled_data)
            train_predict_plot[:, :] = np.nan
            train_predict_plot[look_back:len(train_predict)+look_back, :] = train_predict

            # Shift test predictions for plotting
            test_predict_plot = np.empty_like(scaled_data)
            test_predict_plot[:, :] = np.nan
            # Corrected slicing for test_predict_plot:
            # It should start after the train data, and account for the look_back in test set
            test_predict_plot[len(X_train) + look_back : len(scaled_data), :] = test_predict

            # Prepare plot data (using original full data for actuals)
            plot_data = {
                'test_dates': data.index, # For main prediction, dates should cover the full period
                'actual_values': data['Close'].values, # Actual values for the whole period
                'train_predict': train_predict_plot[:, 0],
                'test_predict': test_predict_plot[:, 0],
                'model_name': model_architecture
            }


            self.master.after(0, self._update_metrics_and_plot,
                              {"MSE": test_mse, "R2": test_r2, "MAPE": test_mape, "Accuracy (0-1)": test_accuracy,
                               "Prediction Period": f"{data.index[train_size + look_back].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                               "Num Predictions": len(Y_test_actual)},
                              plot_data, 'main')

            self.status_label.config(text="Main prediction complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Error occurred during main prediction.")
            self._clear_main_plot()
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)
            self.run_future_button.config(state=tk.NORMAL) # Enable future prediction button too


    def _clear_main_plot(self):
        self.ax_main.clear()
        self.ax_main.set_title('Stock Price Prediction')
        self.ax_main.set_xlabel('Date')
        self.ax_main.set_ylabel('Stock Price')
        self.ax_main.grid(True)
        self.fig_main.tight_layout()
        self.canvas_main.draw()

    def _update_metrics_and_plot(self, metrics, plot_data, tab_type):
        ax = None
        canvas = None
        metric_labels = None
        status_label = None

        if tab_type == 'main':
            ax = self.ax_main
            canvas = self.canvas_main
            metric_labels = self.metric_labels
            status_label = self.status_label
            ax.clear()
            ax.plot(plot_data['dates'], plot_data['actual_values'], label='Actual Values', color='black', linewidth=2)
            ax.plot(plot_data['dates'], plot_data['train_predict'], label=f'{plot_data["model_name"]} Train Predict', color='blue')
            ax.plot(plot_data['dates'], plot_data['test_predict'], label=f'{plot_data["model_name"]} Test Predict', color='red', linestyle='--')
            ax.set_title(f'{self.ticker_var.get()} Actual vs. Predicted Values')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.legend()
            ax.grid(True)
            self.fig_main.tight_layout()
            canvas.draw()

        elif tab_type == 'avg':
            ax = self.ax_avg
            canvas = self.canvas_avg
            metric_labels = self.avg_metric_labels
            status_label = self.avg_status_label
            ax.clear()
            # For averaging, plot only actuals and the averaged test predictions
            ax.plot(plot_data['test_dates'], plot_data['actual_values'], label='Actual Values', color='black', linewidth=2)
            ax.plot(plot_data['test_dates'], plot_data['test_predict'],
                     label=f'{plot_data["model_name"]} Predicted', color='red', linestyle='--')
            ax.set_title(f'{self.avg_ticker_var.get()} Averaging Test: Actual vs. Predicted Values')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.legend()
            ax.grid(True)
            self.fig_avg.tight_layout()
            canvas.draw()


        elif tab_type == 'abs':
            ax = self.ax_abs
            canvas = self.canvas_abs
            metric_labels = self.abs_metric_labels
            status_label = self.abs_status_label
            ax.clear()
            ax.plot(plot_data['test_dates'], plot_data['actual_values'], label='Actual Values', color='black', linewidth=2)
            ax.plot(plot_data['test_dates'], plot_data['test_predict'],
                             label=f'{plot_data["model_name"]} Predicted', color='red', linestyle='--')
            ax.set_title(f'{self.abs_ticker_var.get()} Absolute Test: Actual vs. Predicted Values')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.legend()
            ax.grid(True)
            self.fig_abs.tight_layout()
            canvas.draw()


        elif tab_type == 'future': # NEW: Future Prediction
            ax = self.ax_future
            canvas = self.canvas_future
            metric_labels = self.future_metric_labels
            status_label = self.future_status_label

            ax.clear()
            # Plot historical data leading up to prediction
            ax.plot(plot_data['history_dates'], plot_data['history_values'], label='Historical Values (Train)', color='black', linewidth=2)
            # Plot predicted future values
            ax.plot(plot_data['future_dates'], plot_data['future_predict'], label=f'{plot_data["model_name"]} Predicted Future', color='red', linestyle='--')
            # Add a vertical line to mark the transition from historical to future
            if not plot_data['history_dates'].empty:
                ax.axvline(x=plot_data['history_dates'][-1], color='gray', linestyle=':', label='End of Historical Data / Start of Prediction')
            ax.set_title(f'{self.future_ticker_var.get()} Future Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.legend()
            ax.grid(True)
            self.fig_future.tight_layout()
            canvas.draw()
            # For future prediction, metrics are just info, no accuracy/MSE etc.
            for metric_name, value in metrics.items():
                metric_labels[metric_name].config(text=str(value))
            status_label.config(text="Future prediction complete!")
            return # Exit early as metrics display is different for future prediction

        # Default metric display for main, avg, abs tabs
        for metric_name, value in metrics.items():
            if "MSE" in metric_name or "R2" in metric_name:
                metric_labels[metric_name].config(text=f"{value:.4f}")
            elif "MAPE" in metric_name:
                metric_labels[metric_name].config(text=f"{value:.2f}%")
            elif "Accuracy" in metric_name: # Accuracy is already 0-1, format it as percentage
                 metric_labels[metric_name].config(text=f"{value*100:.2f}%")
            else:
                metric_labels[metric_name].config(text=str(value))

        status_label.config(text="Prediction complete!") # Generic message for other tabs

    # --- Averaging Tab ---
    def _create_averaging_tab(self):
        self.averaging_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.averaging_tab, text="Averaging Prediction")

        # Left Panel (Controls)
        self.left_panel_avg = ttk.LabelFrame(self.averaging_tab, text="Configuration (Averaging)")
        self.left_panel_avg.pack(side="left", fill="y", padx=10, pady=10)

        # Ticker Symbol
        ttk.Label(self.left_panel_avg, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.avg_ticker_var = tk.StringVar(value="GOOGL")
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Start Date
        ttk.Label(self.left_panel_avg, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.avg_start_date_var = tk.StringVar(value="2010-01-01")
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        # End Date
        ttk.Label(self.left_panel_avg, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.avg_end_date_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Look-back Period
        ttk.Label(self.left_panel_avg, text="Look-back Period:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.avg_look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_look_back_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        # Train/Test Split
        ttk.Label(self.left_panel_avg, text="Train/Test Split (%):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.avg_train_test_split_var = tk.IntVar(value=80)
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_train_test_split_var).grid(row=4, column=1, padx=5, pady=2, sticky="ew")

        # Number of Models for Averaging
        ttk.Label(self.left_panel_avg, text="Num Models (Avg):").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.num_avg_models_var = tk.IntVar(value=3)
        ttk.Entry(self.left_panel_avg, textvariable=self.num_avg_models_var).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # LSTM Units
        ttk.Label(self.left_panel_avg, text="LSTM Units:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.avg_lstm_units_var = tk.IntVar(value=50)
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_lstm_units_var).grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        # Number of LSTM Layers
        ttk.Label(self.left_panel_avg, text="LSTM Layers:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.avg_num_lstm_layers_var = tk.IntVar(value=1)
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_num_lstm_layers_var).grid(row=7, column=1, padx=5, pady=2, sticky="ew")

        # Dropout Rate
        ttk.Label(self.left_panel_avg, text="Dropout Rate:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.avg_dropout_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_dropout_rate_var).grid(row=8, column=1, padx=5, pady=2, sticky="ew")

        # Learning Rate
        ttk.Label(self.left_panel_avg, text="Learning Rate:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        self.avg_learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(self.left_panel_avg, textvariable=self.avg_learning_rate_var).grid(row=9, column=1, padx=5, pady=2, sticky="ew")

        # Run Prediction Button
        self.run_avg_button = ttk.Button(self.left_panel_avg, text="Run Averaging Prediction", command=self._start_averaging_prediction_thread)
        self.run_avg_button.grid(row=10, column=0, columnspan=2, pady=10)

        # Status Label
        self.avg_status_label = ttk.Label(self.left_panel_avg, text="Ready.")
        self.avg_status_label.grid(row=11, column=0, columnspan=2, pady=5)

        # Metrics Display
        self.avg_metrics_frame = ttk.LabelFrame(self.left_panel_avg, text="Prediction Metrics")
        self.avg_metrics_frame.grid(row=12, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        self.avg_metric_labels = {}
        metrics = ["MSE", "R2", "MAPE", "Accuracy (0-1)", "Prediction Period", "Num Predictions"]
        for i, metric in enumerate(metrics):
            ttk.Label(self.avg_metrics_frame, text=f"{metric}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.avg_metric_labels[metric] = ttk.Label(self.avg_metrics_frame, text="N/A")
            self.avg_metric_labels[metric].grid(row=i, column=1, padx=5, pady=2, sticky="w")

        # Right Panel (Plot)
        self.right_panel_avg = ttk.Frame(self.averaging_tab)
        self.right_panel_avg.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig_avg, self.ax_avg = plt.subplots(figsize=(8, 6))
        self.canvas_avg = FigureCanvasTkAgg(self.fig_avg, master=self.right_panel_avg)
        self.canvas_avg_widget = self.canvas_avg.get_tk_widget()
        self.canvas_avg_widget.pack(side="top", fill="both", expand=True)

        self.toolbar_avg = NavigationToolbar2Tk(self.canvas_avg_widget, self.right_panel_avg)
        self.toolbar_avg.update()
        self.canvas_avg_widget.pack(side="top", fill="both", expand=True)

    def _start_averaging_prediction_thread(self):
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)
        self.run_future_button.config(state=tk.DISABLED) # Disable future prediction button too

        self.avg_status_label.config(text="Averaging prediction in progress...")
        self.running_thread = threading.Thread(target=self._run_averaging_prediction)
        self.running_thread.start()

    def _run_averaging_prediction(self):
        try:
            ticker = self.avg_ticker_var.get()
            start_date = self.avg_start_date_var.get()
            end_date = self.avg_end_date_var.get()
            look_back = self.avg_look_back_var.get()
            train_test_split_ratio = self.avg_train_test_split_var.get() / 100
            num_avg_models = self.num_avg_models_var.get()
            lstm_units = self.avg_lstm_units_var.get()
            num_lstm_layers = self.avg_num_lstm_layers_var.get()
            dropout_rate = self.avg_dropout_rate_var.get()
            learning_rate = self.avg_learning_rate_var.get()

            # Data Acquisition
            data = get_stock_data_live(ticker, start_date, end_date)
            if data.empty:
                messagebox.showerror("Error", "No data downloaded for averaging prediction.")
                return

            scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['Close'].values.reshape(-1, 1))

            X, Y = create_sequences(scaled_data, look_back)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            train_size = int(len(X) * train_test_split_ratio)
            X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
            Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

            all_test_predictions = []

            for i in range(num_avg_models):
                self.master.after(0, self.avg_status_label.config, {'text': f"Training model {i+1}/{num_avg_models}..."})
                model_architecture = np.random.choice(list(model_builders.keys())) # Randomly select model type for averaging
                model_builder = model_builders[model_architecture]
                model = model_builder(
                    input_shape=(look_back, 1),
                    lstm_units=lstm_units,
                    num_lstm_layers=num_lstm_layers,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
                early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)
                model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=0, callbacks=[early_stopping])
                test_predict = model.predict(X_test)
                all_test_predictions.append(test_predict)

            # Average the predictions
            avg_test_predict = np.mean(np.array(all_test_predictions), axis=0)

            # Invert predictions to original scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data['Close'].values.reshape(-1, 1))

            avg_test_predict = scaler.inverse_transform(avg_test_predict)
            Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

            # Metric calculation
            test_mse = mean_squared_error(Y_test_actual, avg_test_predict)
            test_r2 = r2_score(Y_test_actual, avg_test_predict)
            test_mape = np.mean(np.abs((Y_test_actual - avg_test_predict) / Y_test_actual)) * 100
            test_accuracy = 1 - (np.abs((Y_test_actual - avg_test_predict) / Y_test_actual)).mean()

            # Prepare plot data (using original full data for actuals)
            scaled_data_full = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['Close'].values.reshape(-1, 1))
            train_predict_avg_plot = np.empty_like(scaled_data_full)
            train_predict_avg_plot[:, :] = np.nan
            # For simplicity, we're only plotting avg test, but if we had avg train, we'd add it here.
            # No average train prediction done here, so leave empty for train part

            avg_test_predict_plot = np.empty_like(scaled_data_full)
            avg_test_predict_plot[:, :] = np.nan
            avg_test_predict_plot[len(X_train)+look_back:len(X_train)+look_back+len(avg_test_predict), :] = avg_test_predict

            plot_data = {
                'test_dates': data.index[train_size + look_back:], # Dates for test predictions
                'actual_values': data['Close'].values[train_size + look_back:], # Actuals for test period
                'test_predict': avg_test_predict_plot[train_size + look_back:, 0], # Avg test predictions
                'model_name': "Averaged Models"
            }

            self.master.after(0, self._update_metrics_and_plot,
                              {"MSE": test_mse, "R2": test_r2, "MAPE": test_mape, "Accuracy (0-1)": test_accuracy,
                               "Prediction Period": f"{data.index[train_size + look_back].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                               "Num Predictions": len(Y_test_actual)},
                              plot_data, 'avg')

            self.avg_status_label.config(text="Averaging prediction complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.avg_status_label.config(text="Error occurred during averaging prediction.")
            self._clear_avg_plot()
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)
            self.run_future_button.config(state=tk.NORMAL) # Enable future prediction button too


    def _clear_avg_plot(self):
        self.ax_avg.clear()
        self.ax_avg.set_title('Averaging Prediction')
        self.ax_avg.set_xlabel('Date')
        self.ax_avg.set_ylabel('Stock Price')
        self.ax_avg.grid(True)
        self.fig_avg.tight_layout()
        self.canvas_avg.draw()

    # --- Absolute Testing Tab ---
    def _create_absolute_testing_tab(self):
        self.absolute_testing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.absolute_testing_tab, text="Absolute Testing")

        # Left Panel (Controls)
        # Fix: self.abs_config_frame needs to be defined within this method's scope or passed.
        # Moving this frame definition to a correct parent.
        self.left_panel_absolute = ttk.LabelFrame(self.absolute_testing_tab, text="Absolute Test Configuration")
        self.left_panel_absolute.pack(side="left", fill="y", padx=10, pady=10)


        # Ticker Symbol
        ttk.Label(self.left_panel_absolute, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.abs_ticker_var = tk.StringVar(value="GOOGL")
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Train Data Start Date
        ttk.Label(self.left_panel_absolute, text="Train Data Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.abs_train_start_date_var = tk.StringVar(value="2005-01-01")
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_train_start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        # Train Data End Date
        ttk.Label(self.left_panel_absolute, text="Train Data End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.abs_train_end_date_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_train_end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Prediction Start Date (for unseen data)
        ttk.Label(self.left_panel_absolute, text="Prediction Start Date (YYYY-MM-DD):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.abs_pred_start_date_var = tk.StringVar(value="2024-01-01")
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_pred_start_date_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        # Prediction End Date (for unseen data)
        ttk.Label(self.left_panel_absolute, text="Prediction End Date (YYYY-MM-DD):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.abs_pred_end_date_var = tk.StringVar(value="2025-12-31")
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_pred_end_date_var).grid(row=4, column=1, padx=5, pady=2, sticky="ew")

        # Look-back Period
        ttk.Label(self.left_panel_absolute, text="Look-back Period:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.abs_look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_look_back_var).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # Model Architecture
        ttk.Label(self.left_panel_absolute, text="Model Architecture:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.abs_model_architecture_var = tk.StringVar(value="LSTM")
        ttk.OptionMenu(self.left_panel_absolute, self.abs_model_architecture_var, "LSTM", *model_builders.keys()).grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        # LSTM Units
        ttk.Label(self.left_panel_absolute, text="LSTM Units:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.abs_lstm_units_var = tk.IntVar(value=50)
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_lstm_units_var).grid(row=7, column=1, padx=5, pady=2, sticky="ew")

        # Number of LSTM Layers
        ttk.Label(self.left_panel_absolute, text="LSTM Layers:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.abs_num_lstm_layers_var = tk.IntVar(value=1)
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_num_lstm_layers_var).grid(row=8, column=1, padx=5, pady=2, sticky="ew")

        # Dropout Rate
        ttk.Label(self.left_panel_absolute, text="Dropout Rate:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        self.abs_dropout_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_dropout_rate_var).grid(row=9, column=1, padx=5, pady=2, sticky="ew")

        # Learning Rate
        ttk.Label(self.left_panel_absolute, text="Learning Rate:").grid(row=10, column=0, padx=5, pady=2, sticky="w")
        self.abs_learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(self.left_panel_absolute, textvariable=self.abs_learning_rate_var).grid(row=10, column=1, padx=5, pady=2, sticky="ew")

        # Run Absolute Test Button
        self.run_abs_button = ttk.Button(self.left_panel_absolute, text="Run Absolute Test", command=self._start_absolute_test_thread)
        self.run_abs_button.grid(row=11, column=0, columnspan=2, pady=10)

        # Status Label
        self.abs_status_label = ttk.Label(self.left_panel_absolute, text="Ready.")
        self.abs_status_label.grid(row=12, column=0, columnspan=2, pady=5)

        # Metrics Display
        self.abs_metrics_frame = ttk.LabelFrame(self.left_panel_absolute, text="Absolute Test Metrics")
        self.abs_metrics_frame.grid(row=13, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        self.abs_metric_labels = {}
        metrics = ["MSE", "R2", "MAPE", "Accuracy (0-1)", "Prediction Period", "Num Predictions"]
        for i, metric in enumerate(metrics):
            ttk.Label(self.abs_metrics_frame, text=f"{metric}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.abs_metric_labels[metric] = ttk.Label(self.abs_metrics_frame, text="N/A")
            self.abs_metric_labels[metric].grid(row=i, column=1, padx=5, pady=2, sticky="w")

        # Right Panel (Plot)
        self.right_panel_absolute = ttk.Frame(self.absolute_testing_tab)
        self.right_panel_absolute.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig_abs, self.ax_abs = plt.subplots(figsize=(8, 6))
        self.canvas_abs = FigureCanvasTkAgg(self.fig_abs, master=self.right_panel_absolute)
        self.canvas_abs_widget = self.canvas_abs.get_tk_widget()
        self.canvas_abs_widget.pack(side="top", fill="both", expand=True)

        self.toolbar_abs = NavigationToolbar2Tk(self.canvas_abs_widget, self.right_panel_absolute)
        self.toolbar_abs.update()
        self.canvas_abs_widget.pack(side="top", fill="both", expand=True)


    def _start_absolute_test_thread(self):
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)
        self.run_future_button.config(state=tk.DISABLED) # Disable future prediction button too

        self.abs_status_label.config(text="Absolute test in progress...")
        self.running_thread = threading.Thread(target=self._run_absolute_test)
        self.running_thread.start()

    def _run_absolute_test(self):
        try:
            ticker = self.abs_ticker_var.get()
            train_start_date = self.abs_train_start_date_var.get()
            train_end_date = self.abs_train_end_date_var.get()
            pred_start_date = self.abs_pred_start_date_var.get()
            pred_end_date = self.abs_pred_end_date_var.get()
            look_back = self.abs_look_back_var.get()
            model_architecture = self.abs_model_architecture_var.get()
            lstm_units = self.abs_lstm_units_var.get()
            num_lstm_layers = self.abs_num_lstm_layers_var.get()
            dropout_rate = self.abs_dropout_rate_var.get()
            learning_rate = self.abs_learning_rate_var.get()

            # Data Acquisition
            # Train data: Fetch only the training period
            data_train = get_stock_data_live(ticker, train_start_date, train_end_date)
            if data_train.empty:
                messagebox.showerror("Error", "No training data downloaded for absolute test.")
                self._clear_abs_plot()
                return

            # Prediction data: Fetch only the prediction period
            data_pred = get_stock_data_live(ticker, pred_start_date, pred_end_date)
            if data_pred.empty:
                messagebox.showerror("Warning", "No prediction data downloaded for absolute test. Cannot calculate metrics.")
                self._clear_abs_plot()
                return


            # Scaling and sequence creation for training data
            scaler_train = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler_train.fit_transform(data_train['Close'].values.reshape(-1, 1))
            X_train, Y_train = create_sequences(scaled_train_data, look_back)

            # Reshape for LSTM
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Model Building and Training
            model_builder = model_builders[model_architecture]
            model = model_builder(
                input_shape=(look_back, 1),
                lstm_units=lstm_units,
                num_lstm_layers=num_lstm_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )

            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])

            # Prepare data for prediction (this is the crucial part)
            # We need the last `look_back` days from the *training data* to make the first prediction
            # for the *prediction period*.
            last_train_sequence = scaled_train_data[-look_back:]

            # To predict the entire prediction period, we need to do it iteratively,
            # feeding the model its own predictions.

            # Initialize list to store predictions
            predicted_scaled_values = []
            current_sequence = last_train_sequence.copy()

            # Generate predictions day by day
            for i in range(len(data_pred)):
                # Reshape for model input
                input_seq = current_sequence.reshape(1, look_back, 1)
                # Predict next value
                next_predicted_value_scaled = model.predict(input_seq)[0, 0]
                predicted_scaled_values.append(next_predicted_value_scaled)

                # Update current_sequence: remove the oldest value, add the new predicted value
                current_sequence = np.append(current_sequence[1:], [[next_predicted_value_scaled]], axis=0)

            # Invert predictions to original scale
            test_predict = scaler_train.inverse_transform(np.array(predicted_scaled_values).reshape(-1, 1))

            # Align actual values from data_pred
            # CORRECTED: Get Y_actual as a 1D array for proper comparison
            Y_actual = data_pred['Close'].values

            # Ensure lengths match for metrics calculation (if data_pred was shorter than expected)
            min_len = min(len(Y_actual), len(test_predict))
            Y_actual = Y_actual[:min_len]
            test_predict = test_predict[:min_len]
            pred_dates = data_pred.index[:min_len]

            # Metric calculation - Corrected to compare full series
            mse = mean_squared_error(Y_actual, test_predict[:, 0])
            r2 = r2_score(Y_actual, test_predict[:, 0])
            # Check for division by zero in MAPE if Y_actual contains zeros
            mape = np.mean(np.abs((Y_actual - test_predict[:, 0]) / np.where(Y_actual == 0, 1e-8, Y_actual))) * 100
            accuracy = 1 - (np.abs((Y_actual - test_predict[:, 0]) / np.where(Y_actual == 0, 1e-8, Y_actual))).mean()


            # Prepare plot data (using original full data for actuals)
            plot_data = {
                'test_dates': pred_dates,
                'actual_values': Y_actual, # Corrected to pass the full array
                'test_predict': test_predict[:, 0],
                'model_name': model_architecture
            }

            self.master.after(0, self._update_metrics_and_plot,
                              {"MSE": mse, "R2": r2, "MAPE": mape, "Accuracy (0-1)": accuracy,
                               "Prediction Period": f"{pred_start_date} to {pred_end_date}",  # Show requested period
                               "Num Predictions": len(plot_data['test_dates'])},
                              plot_data, 'abs')

            self.abs_status_label.config(text="Absolute test complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.abs_status_label.config(text="Error occurred during absolute testing.")
            self._clear_abs_plot()
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)
            self.run_future_button.config(state=tk.NORMAL) # Enable future prediction button too


    def _clear_abs_plot(self):
        self.ax_abs.clear()
        self.ax_abs.set_title('Absolute Test Prediction')
        self.ax_abs.set_xlabel('Date')
        self.ax_abs.set_ylabel('Stock Price')
        self.ax_abs.grid(True)
        self.fig_abs.tight_layout()
        self.canvas_abs.draw()

    # --- Hyperparameter Experiment Tab ---
    def _create_experiment_tab(self):
        self.experiment_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.experiment_tab, text="Hyperparameter Experiment")

        # Left Panel (Configuration)
        self.left_panel_experiment = ttk.LabelFrame(self.experiment_tab, text="Experiment Configuration")
        self.left_panel_experiment.pack(side="left", fill="y", padx=10, pady=10)

        # Ticker
        ttk.Label(self.left_panel_experiment, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.exp_ticker_var = tk.StringVar(value="GOOGL")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Start Date
        ttk.Label(self.left_panel_experiment, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.exp_start_date_var = tk.StringVar(value="2010-01-01")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        # End Date
        ttk.Label(self.left_panel_experiment, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.exp_end_date_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Look-back Period
        ttk.Label(self.left_panel_experiment, text="Look-back (comma-sep):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.exp_look_back_var = tk.StringVar(value="30,60")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_look_back_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        # LSTM Units
        ttk.Label(self.left_panel_experiment, text="LSTM Units (comma-sep):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.exp_lstm_units_var = tk.StringVar(value="50,100")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_lstm_units_var).grid(row=4, column=1, padx=5, pady=2, sticky="ew")

        # Num LSTM Layers
        ttk.Label(self.left_panel_experiment, text="LSTM Layers (comma-sep):").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.exp_num_lstm_layers_var = tk.StringVar(value="1,2")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_num_lstm_layers_var).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # Dropout Rate
        ttk.Label(self.left_panel_experiment, text="Dropout Rate (comma-sep):").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.exp_dropout_rate_var = tk.StringVar(value="0.1,0.2")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_dropout_rate_var).grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        # Learning Rate
        ttk.Label(self.left_panel_experiment, text="Learning Rate (comma-sep):").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.exp_learning_rate_var = tk.StringVar(value="0.001,0.01")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_learning_rate_var).grid(row=7, column=1, padx=5, pady=2, sticky="ew")

        # Model Architectures
        ttk.Label(self.left_panel_experiment, text="Model Architectures (comma-sep):").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.exp_model_arch_var = tk.StringVar(value="LSTM,Bidirectional LSTM")
        ttk.Entry(self.left_panel_experiment, textvariable=self.exp_model_arch_var).grid(row=8, column=1, padx=5, pady=2, sticky="ew")

        # Run Experiment Button
        self.run_experiment_button = ttk.Button(self.left_panel_experiment, text="Run Experiment", command=self._start_experiment_thread)
        self.run_experiment_button.grid(row=9, column=0, columnspan=2, pady=10)

        # Status Label
        self.experiment_status_label = ttk.Label(self.left_panel_experiment, text="Ready.")
        self.experiment_status_label.grid(row=10, column=0, columnspan=2, pady=5)

        # Results Treeview
        self.results_frame = ttk.LabelFrame(self.left_panel_experiment, text="Experiment Results")
        self.results_frame.grid(row=11, column=0, columnspan=2, padx=5, pady=10, sticky="nsew")

        self.tree = ttk.Treeview(self.results_frame, columns=("MSE", "R2", "MAPE", "Accuracy"), show="headings")
        self.tree.heading("MSE", text="MSE")
        self.tree.heading("R2", text="R2")
        self.tree.heading("MAPE", text="MAPE")
        self.tree.heading("Accuracy", text="Accuracy")
        self.tree.pack(fill="both", expand=True)

        # Scrollbar for treeview
        vsb = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=vsb.set)

    def _start_experiment_thread(self):
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)
        self.run_future_button.config(state=tk.DISABLED) # Disable future prediction button too

        self.experiment_status_label.config(text="Experiment in progress...")
        for i in self.tree.get_children(): # Clear previous results
            self.tree.delete(i)
        self.running_thread = threading.Thread(target=self._run_experiment)
        self.running_thread.start()

    def _run_experiment(self):
        try:
            ticker = self.exp_ticker_var.get()
            start_date = self.exp_start_date_var.get()
            end_date = self.exp_end_date_var.get()
            train_test_split_ratio = 0.8 # Fixed for experiments

            # Parse comma-separated hyperparameters
            look_backs = [int(x) for x in self.exp_look_back_var.get().split(',') if x.strip()]
            lstm_units_options = [int(x) for x in self.exp_lstm_units_var.get().split(',') if x.strip()]
            num_lstm_layers_options = [int(x) for x in self.exp_num_lstm_layers_var.get().split(',') if x.strip()]
            dropout_rates = [float(x) for x in self.exp_dropout_rate_var.get().split(',') if x.strip()]
            learning_rates = [float(x) for x in self.exp_learning_rate_var.get().split(',') if x.strip()]
            model_architectures = [x.strip() for x in self.exp_model_arch_var.get().split(',') if x.strip()]

            # Generate all combinations of hyperparameters
            hyperparameter_combinations = list(itertools.product(
                look_backs, lstm_units_options, num_lstm_layers_options,
                dropout_rates, learning_rates, model_architectures
            ))

            # Data Acquisition (once for all experiments)
            data = get_stock_data_live(ticker, start_date, end_date)
            if data.empty:
                messagebox.showerror("Error", "No data downloaded for experiment.")
                return

            # Main loop for experiments
            for i, (look_back, lstm_units, num_lstm_layers, dropout_rate, learning_rate, model_architecture) in enumerate(hyperparameter_combinations):
                self.master.after(0, self.experiment_status_label.config, {'text': f"Running experiment {i+1}/{len(hyperparameter_combinations)}..."})

                # Data preparation (inside loop as look_back changes)
                scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data['Close'].values.reshape(-1, 1))
                X, Y = create_sequences(scaled_data, look_back)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                train_size = int(len(X) * train_test_split_ratio)
                X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
                Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

                # Model Building and Training
                model_builder = model_builders[model_architecture]
                model = model_builder(
                    input_shape=(look_back, 1),
                    lstm_units=lstm_units,
                    num_lstm_layers=num_lstm_layers,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )

                early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)
                model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=0, callbacks=[early_stopping])

                # Making predictions on the test set
                test_predict = model.predict(X_test)

                # Invert predictions to original scale
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler.fit(data['Close'].values.reshape(-1, 1)) # Fit scaler on original full data range

                test_predict = scaler.inverse_transform(test_predict)
                Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

                # Metric calculation
                mse = mean_squared_error(Y_test_actual, test_predict)
                r2 = r2_score(Y_test_actual, test_predict)
                mape = np.mean(np.abs((Y_test_actual - test_predict) / np.where(Y_test_actual == 0, 1e-8, Y_test_actual))) * 100
                accuracy = 1 - (np.abs((Y_test_actual - test_predict) / np.where(Y_test_actual == 0, 1e-8, Y_test_actual))).mean()

                # Add result to treeview
                self.master.after(0, self.tree.insert, "", "end", values=(
                    f"LB:{look_back}, LU:{lstm_units}, LL:{num_lstm_layers}, DR:{dropout_rate}, LR:{learning_rate}, Arch:{model_architecture}",
                    f"{mse:.4f}", f"{r2:.4f}", f"{mape:.2f}%", f"{accuracy*100:.2f}%" # Format accuracy as percentage
                ))
            self.experiment_status_label.config(text="Experiment complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.experiment_status_label.config(text="Error occurred during experiment.")
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)
            self.run_future_button.config(state=tk.NORMAL) # Enable future prediction button too

    # --- NEW: Future Prediction Tab ---
    def _create_future_prediction_tab(self):
        self.future_prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.future_prediction_tab, text="Future Prediction")

        # Left Panel (Controls)
        self.left_panel_future = ttk.LabelFrame(self.future_prediction_tab, text="Future Prediction Configuration")
        self.left_panel_future.pack(side="left", fill="y", padx=10, pady=10)

        # Ticker Symbol
        ttk.Label(self.left_panel_future, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.future_ticker_var = tk.StringVar(value="GOOGL")
        ttk.Entry(self.left_panel_future, textvariable=self.future_ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Prediction Start Date
        ttk.Label(self.left_panel_future, text="Prediction Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        # Default to tomorrow's date
        tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        self.future_pred_start_date_var = tk.StringVar(value=tomorrow)
        ttk.Entry(self.left_panel_future, textvariable=self.future_pred_start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        # Prediction End Date
        ttk.Label(self.left_panel_future, text="Prediction End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        # Default to 1 year from tomorrow
        one_year_from_tomorrow = (datetime.date.today() + datetime.timedelta(days=366)).strftime("%Y-%m-%d")
        self.future_pred_end_date_var = tk.StringVar(value=one_year_from_tomorrow)
        ttk.Entry(self.left_panel_future, textvariable=self.future_pred_end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Look-back Period
        ttk.Label(self.left_panel_future, text="Look-back Period:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.future_look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.left_panel_future, textvariable=self.future_look_back_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        # Model Architecture
        ttk.Label(self.left_panel_future, text="Model Architecture:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.future_model_architecture_var = tk.StringVar(value="LSTM")
        ttk.OptionMenu(self.left_panel_future, self.future_model_architecture_var, "LSTM", *model_builders.keys()).grid(row=4, column=1, padx=5, pady=2, sticky="ew")

        # LSTM Units
        ttk.Label(self.left_panel_future, text="LSTM Units:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.future_lstm_units_var = tk.IntVar(value=50)
        ttk.Entry(self.left_panel_future, textvariable=self.future_lstm_units_var).grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # Number of LSTM Layers
        ttk.Label(self.left_panel_future, text="LSTM Layers:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.future_num_lstm_layers_var = tk.IntVar(value=1)
        ttk.Entry(self.left_panel_future, textvariable=self.future_num_lstm_layers_var).grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        # Dropout Rate
        ttk.Label(self.left_panel_future, text="Dropout Rate:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.future_dropout_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.left_panel_future, textvariable=self.future_dropout_rate_var).grid(row=7, column=1, padx=5, pady=2, sticky="ew")

        # Learning Rate
        ttk.Label(self.left_panel_future, text="Learning Rate:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.future_learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(self.left_panel_future, textvariable=self.future_learning_rate_var).grid(row=8, column=1, padx=5, pady=2, sticky="ew")

        # Run Future Prediction Button
        self.run_future_button = ttk.Button(self.left_panel_future, text="Run Future Prediction", command=self._start_future_prediction_thread)
        self.run_future_button.grid(row=9, column=0, columnspan=2, pady=10)

        # Status Label
        self.future_status_label = ttk.Label(self.left_panel_future, text="Ready.")
        self.future_status_label.grid(row=10, column=0, columnspan=2, pady=5)

        # Metrics Display (for future, it will be limited as no actuals)
        self.future_metrics_frame = ttk.LabelFrame(self.left_panel_future, text="Future Prediction Info")
        self.future_metrics_frame.grid(row=11, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        self.future_metric_labels = {}
        # Metrics will be different here, mostly info about the prediction
        metrics = ["Prediction Period", "Num Predictions"]
        for i, metric in enumerate(metrics):
            ttk.Label(self.future_metrics_frame, text=f"{metric}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.future_metric_labels[metric] = ttk.Label(self.future_metrics_frame, text="N/A")
            self.future_metric_labels[metric].grid(row=i, column=1, padx=5, pady=2, sticky="w")

        # Right Panel (Plot)
        self.right_panel_future = ttk.Frame(self.future_prediction_tab)
        self.right_panel_future.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig_future, self.ax_future = plt.subplots(figsize=(8, 6))
        self.canvas_future = FigureCanvasTkAgg(self.fig_future, master=self.right_panel_future)
        self.canvas_future_widget = self.canvas_future.get_tk_widget()
        self.canvas_future_widget.pack(side="top", fill="both", expand=True)

        self.toolbar_future = NavigationToolbar2Tk(self.canvas_future_widget, self.right_panel_future)
        self.toolbar_future.update()
        self.canvas_future_widget.pack(side="top", fill="both", expand=True)

    def _start_future_prediction_thread(self):
        # Disable all relevant buttons while prediction is running
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)
        self.run_future_button.config(state=tk.DISABLED)

        self.future_status_label.config(text="Future prediction in progress...")
        self.running_thread = threading.Thread(target=self._run_future_prediction)
        self.running_thread.start()

    def _run_future_prediction(self):
        try:
            ticker = self.future_ticker_var.get()
            pred_start_date_str = self.future_pred_start_date_var.get()
            pred_end_date_str = self.future_pred_end_date_var.get()
            look_back = self.future_look_back_var.get()
            model_architecture = self.future_model_architecture_var.get()
            lstm_units = self.future_lstm_units_var.get()
            num_lstm_layers = self.future_num_lstm_layers_var.get()
            dropout_rate = self.future_dropout_rate_var.get()
            learning_rate = self.future_learning_rate_var.get()

            # Convert date strings to datetime objects for calculations
            pred_start_date = datetime.datetime.strptime(pred_start_date_str, "%Y-%m-%d").date()
            pred_end_date = datetime.datetime.strptime(pred_end_date_str, "%Y-%m-%d").date()

            # Determine training data end date: up to the day before future prediction starts
            # If future_pred_start_date is today or earlier, adjust to yesterday's date
            # Ensure the training data end date is always before the prediction start date.
            train_end_date_dt = pred_start_date - datetime.timedelta(days=1)
            # Ensure train_end_date is not in the future for yfinance download
            if train_end_date_dt > datetime.date.today():
                train_end_date_dt = datetime.date.today() - datetime.timedelta(days=1)
                if train_end_date_dt < datetime.date(2000,1,1): # Ensure it doesn't go too far back
                    train_end_date_dt = datetime.date(2000,1,1) # Set a reasonable floor

            # Fetch historical data from a very early date up to the calculated train_end_date
            # A fixed early date (e.g., 2000-01-01) ensures comprehensive historical training.
            train_start_date = "2000-01-01"

            # Data Acquisition - Training on all available historical data
            data_train = get_stock_data_live(ticker, train_start_date, train_end_date_dt.strftime("%Y-%m-%d"))
            if data_train.empty:
                messagebox.showerror("Error", "No historical data downloaded for future prediction training. Adjust dates or check ticker.")
                self._clear_future_plot()
                return

            # Scaling and sequence creation for training data
            scaler_train = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler_train.fit_transform(data_train['Close'].values.reshape(-1, 1))
            X_train, Y_train = create_sequences(scaled_train_data, look_back)

            # Reshape for LSTM
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Model Building and Training
            model_builder = model_builders[model_architecture]
            model = model_builder(
                input_shape=(look_back, 1),
                lstm_units=lstm_units,
                num_lstm_layers=num_lstm_layers,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )

            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])

            # Generate future prediction dates
            # Create a date range for the future prediction
            future_dates = pd.bdate_range(start=pred_start_date, end=pred_end_date)
            if future_dates.empty:
                messagebox.showerror("Error", "No valid business days in the specified future prediction range.")
                self._clear_future_plot()
                return

            # Prepare data for prediction - crucial: use last `look_back` days from *training data*
            last_train_sequence = scaled_train_data[-look_back:]

            # Initialize list to store future predictions
            predicted_scaled_future_values = []
            current_sequence = last_train_sequence.copy()

            # Generate predictions day by day for the future period
            for i in range(len(future_dates)):
                input_seq = current_sequence.reshape(1, look_back, 1)
                next_predicted_value_scaled = model.predict(input_seq)[0, 0]
                predicted_scaled_future_values.append(next_predicted_value_scaled)

                # Update current_sequence with the new prediction
                current_sequence = np.append(current_sequence[1:], [[next_predicted_value_scaled]], axis=0)

            # Invert predictions to original scale
            future_predict = scaler_train.inverse_transform(np.array(predicted_scaled_future_values).reshape(-1, 1))

            # Prepare plot data for future prediction
            plot_data = {
                'history_dates': data_train.index,
                'history_values': data_train['Close'].values,
                'future_dates': future_dates,
                'future_predict': future_predict[:, 0],
                'model_name': model_architecture
            }

            self.master.after(0, self._update_metrics_and_plot,
                              {"Prediction Period": f"{pred_start_date_str} to {pred_end_date_str}",
                               "Num Predictions": len(future_dates)},
                              plot_data, 'future')

            self.future_status_label.config(text="Future prediction complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.future_status_label.config(text="Error occurred during future prediction.")
            self._clear_future_plot()
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)
            self.run_future_button.config(state=tk.NORMAL)

    def _clear_future_plot(self):
        self.ax_future.clear()
        self.ax_future.set_title('Future Stock Price Prediction')
        self.ax_future.set_xlabel('Date')
        self.ax_future.set_ylabel('Stock Price')
        self.ax_future.grid(True)
        self.fig_future.tight_layout()
        self.canvas_future.draw()


def main():
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()