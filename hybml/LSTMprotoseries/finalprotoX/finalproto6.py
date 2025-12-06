import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import threading
import os
import itertools  # For iterating through combinations of hyperparameters

# Suppress TensorFlow warnings (optional, but can clean up console output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- Data Acquisition ---
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
    if len(data) <= look_back:
        return np.array([]), np.array([])

    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


# --- Model Definitions (Updated to accept hyperparameters) ---
def build_basic_lstm_model(input_shape, lstm_units, num_lstm_layers, dropout_rate, learning_rate):
    model = Sequential()
    for i in range(num_lstm_layers):
        if i == 0:  # First layer needs input_shape
            model.add(LSTM(lstm_units, return_sequences=(num_lstm_layers > 1), input_shape=input_shape))
        else:
            model.add(LSTM(lstm_units, return_sequences=(i < num_lstm_layers - 1)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


def build_bidirectional_lstm_model(input_shape, lstm_units, num_lstm_layers, dropout_rate, learning_rate):
    model = Sequential()
    for i in range(num_lstm_layers):
        if i == 0:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=(num_lstm_layers > 1)), input_shape=input_shape))
        else:
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=(i < num_lstm_layers - 1))))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


def build_conv1d_lstm_model(input_shape, lstm_units, num_lstm_layers, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    for i in range(num_lstm_layers):
        # Only the last LSTM layer should return_sequences=False
        model.add(LSTM(lstm_units, return_sequences=(i < num_lstm_layers - 1)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model


model_builders = {
    "Basic LSTM": build_basic_lstm_model,
    "Bidirectional LSTM": build_bidirectional_lstm_model,
    "Conv1D + LSTM": build_conv1d_lstm_model
}


class StockPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("LSTM Stock Price Predictor")
        master.geometry("1200x800")  # Increased width for new experiment params
        master.resizable(True, True)

        # --- Main Notebook (Tabbed Interface) ---
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Tab 1: Single Prediction ---
        self.single_prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_prediction_tab, text="Single Prediction")
        self.single_prediction_tab.grid_columnconfigure(0, weight=1)
        self.single_prediction_tab.grid_columnconfigure(1, weight=2)
        self.single_prediction_tab.grid_rowconfigure(0, weight=1)

        # Left side of Single Prediction Tab (Config + Metrics)
        self.left_panel_single = ttk.Frame(self.single_prediction_tab)
        self.left_panel_single.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_panel_single.grid_rowconfigure(0, weight=0)
        self.left_panel_single.grid_rowconfigure(1, weight=1)

        # --- Configuration LabelFrame (Single Prediction Tab) ---
        self.config_frame = ttk.LabelFrame(self.left_panel_single, text="Prediction Configuration")
        self.config_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.config_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.config_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.ticker_var = tk.StringVar(value="^GSPC")
        ttk.Entry(self.config_frame, textvariable=self.ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.config_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        today = datetime.date.today()
        ten_years_ago = (today - datetime.timedelta(days=365 * 10)).strftime('%Y-%m-%d')
        self.start_date_var = tk.StringVar(value=ten_years_ago)
        ttk.Entry(self.config_frame, textvariable=self.start_date_var).grid(row=1, column=1, padx=5, pady=2,
                                                                            sticky="ew")

        ttk.Label(self.config_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        yesterday = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        self.end_date_var = tk.StringVar(value=yesterday)
        ttk.Entry(self.config_frame, textvariable=self.end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.config_frame, text="Look Back Window (days):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.config_frame, textvariable=self.look_back_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.config_frame, text="Train/Test Ratio:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.train_ratio_var = tk.DoubleVar(value=0.9)
        self.train_ratio_slider = ttk.Scale(self.config_frame, from_=0.1, to=0.99, orient="horizontal",
                                            variable=self.train_ratio_var, command=self._update_train_ratio_label)
        self.train_ratio_slider.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        self.train_ratio_label = ttk.Label(self.config_frame, text=f"{self.train_ratio_var.get():.2f}")
        self.train_ratio_label.grid(row=4, column=2, padx=5, pady=2, sticky="w")

        ttk.Label(self.config_frame, text="Model Architecture:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.model_var = tk.StringVar(value="Conv1D + LSTM")
        self.model_dropdown = ttk.Combobox(self.config_frame, textvariable=self.model_var,
                                           values=list(model_builders.keys()), state="readonly")
        self.model_dropdown.grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # Fixed hyperparameters for single run
        ttk.Label(self.config_frame, text="LSTM Units:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.single_lstm_units_var = tk.IntVar(value=50)
        ttk.Entry(self.config_frame, textvariable=self.single_lstm_units_var).grid(row=6, column=1, padx=5, pady=2,
                                                                                   sticky="ew")

        ttk.Label(self.config_frame, text="LSTM Layers:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.single_lstm_layers_var = tk.IntVar(value=1)
        ttk.Entry(self.config_frame, textvariable=self.single_lstm_layers_var).grid(row=7, column=1, padx=5, pady=2,
                                                                                    sticky="ew")

        ttk.Label(self.config_frame, text="Dropout Rate:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.single_dropout_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.config_frame, textvariable=self.single_dropout_rate_var).grid(row=8, column=1, padx=5, pady=2,
                                                                                     sticky="ew")

        ttk.Label(self.config_frame, text="Learning Rate:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        self.single_learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(self.config_frame, textvariable=self.single_learning_rate_var).grid(row=9, column=1, padx=5, pady=2,
                                                                                      sticky="ew")

        self.run_button = ttk.Button(self.config_frame, text="Run Single Prediction",
                                     command=self._run_single_prediction_thread)
        self.run_button.grid(row=10, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        self.status_label = ttk.Label(self.config_frame, text="Ready.")
        self.status_label.grid(row=11, column=0, columnspan=3, padx=5, pady=2, sticky="w")

        # --- Metrics Frame (Single Prediction Tab) ---
        self.metrics_frame = ttk.LabelFrame(self.left_panel_single, text="Performance Metrics")
        self.metrics_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.metrics_frame.grid_columnconfigure(1, weight=1)

        self.metric_labels = {}
        metrics_order = ["MSE", "R2", "MAPE", "Accuracy (0-1)", "Prediction Period", "Num Predictions"]
        for i, metric_name in enumerate(metrics_order):
            ttk.Label(self.metrics_frame, text=f"{metric_name}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.metric_labels[metric_name] = ttk.Label(self.metrics_frame, text="N/A")
            self.metric_labels[metric_name].grid(row=i, column=1, padx=5, pady=2, sticky="ew")

        # --- Main Prediction Plot (Single Prediction Tab) ---
        self.main_plot_frame = ttk.LabelFrame(self.single_prediction_tab, text="Single Prediction Plot")
        self.main_plot_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.main_plot_frame.grid_columnconfigure(0, weight=1)
        self.main_plot_frame.grid_rowconfigure(0, weight=1)

        self.fig_main, self.ax_main = plt.subplots(figsize=(10, 6))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=self.main_plot_frame)
        self.canvas_main_widget = self.canvas_main.get_tk_widget()
        self.canvas_main_widget.grid(row=0, column=0, sticky="nsew")

        self.toolbar_main_frame = ttk.Frame(self.main_plot_frame)
        self.toolbar_main_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar_main = NavigationToolbar2Tk(self.canvas_main, self.toolbar_main_frame)
        self.toolbar_main.update()

        self.annot_main = self.ax_main.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                                                arrowprops=dict(arrowstyle="->"))
        self.annot_main.set_visible(False)
        self.canvas_main.mpl_connect("motion_notify_event", self._hover_main_plot)
        self._clear_main_plot()

        # --- Tab 2: Average Single Run ---
        self.average_run_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.average_run_tab, text="Average Single Run")
        # Configure columns for the split layout
        self.average_run_tab.grid_columnconfigure(0, weight=1)  # Left panel for config and metrics
        self.average_run_tab.grid_columnconfigure(1, weight=2)  # Right panel for plot
        self.average_run_tab.grid_rowconfigure(0, weight=1)  # Row for left panel

        # Left side of Average Single Run Tab (Config + Metrics)
        self.left_panel_avg_run = ttk.Frame(self.average_run_tab)
        self.left_panel_avg_run.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_panel_avg_run.grid_rowconfigure(0, weight=0)  # Config frame (shrink to fit content)
        self.left_panel_avg_run.grid_rowconfigure(1, weight=1)  # Metrics frame (expand vertically)

        # --- Configuration LabelFrame (Average Single Run Tab) ---
        self.avg_run_config_frame = ttk.LabelFrame(self.left_panel_avg_run, text="Average Run Configuration")
        self.avg_run_config_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.avg_run_config_frame.grid_columnconfigure(1, weight=1)

        # Re-using default values, but variables are distinct
        ttk.Label(self.avg_run_config_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_ticker_var = tk.StringVar(value="^GSPC")
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_ticker_var).grid(row=0, column=1, padx=5, pady=2,
                                                                                        sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2,
                                                                                   sticky="w")
        self.avg_run_start_date_var = tk.StringVar(value=ten_years_ago)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_start_date_var).grid(row=1, column=1, padx=5,
                                                                                            pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2,
                                                                                 sticky="w")
        self.avg_run_end_date_var = tk.StringVar(value=yesterday)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_end_date_var).grid(row=2, column=1, padx=5,
                                                                                          pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Look Back Window (days):").grid(row=3, column=0, padx=5, pady=2,
                                                                                   sticky="w")
        self.avg_run_look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_look_back_var).grid(row=3, column=1, padx=5,
                                                                                           pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Train/Test Ratio:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_train_ratio_var = tk.DoubleVar(value=0.9)
        self.avg_run_train_ratio_slider = ttk.Scale(self.avg_run_config_frame, from_=0.1, to=0.99, orient="horizontal",
                                                    variable=self.avg_run_train_ratio_var,
                                                    command=self._update_avg_run_train_ratio_label)
        self.avg_run_train_ratio_slider.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        self.avg_run_train_ratio_label = ttk.Label(self.avg_run_config_frame,
                                                   text=f"{self.avg_run_train_ratio_var.get():.2f}")
        self.avg_run_train_ratio_label.grid(row=4, column=2, padx=5, pady=2, sticky="w")

        ttk.Label(self.avg_run_config_frame, text="Model Architecture:").grid(row=5, column=0, padx=5, pady=2,
                                                                              sticky="w")
        self.avg_run_model_var = tk.StringVar(value="Conv1D + LSTM")
        self.avg_run_model_dropdown = ttk.Combobox(self.avg_run_config_frame, textvariable=self.avg_run_model_var,
                                                   values=list(model_builders.keys()), state="readonly")
        self.avg_run_model_dropdown.grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        # Hyperparameters for average run tab
        ttk.Label(self.avg_run_config_frame, text="LSTM Units:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_lstm_units_var = tk.IntVar(value=50)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_lstm_units_var).grid(row=6, column=1, padx=5,
                                                                                            pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="LSTM Layers:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_lstm_layers_var = tk.IntVar(value=1)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_lstm_layers_var).grid(row=7, column=1, padx=5,
                                                                                             pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Dropout Rate:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_dropout_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_dropout_rate_var).grid(row=8, column=1, padx=5,
                                                                                              pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Learning Rate:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_learning_rate_var).grid(row=9, column=1, padx=5,
                                                                                               pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Number of Runs:").grid(row=10, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_num_runs_var = tk.IntVar(value=10)  # Default to 10 runs for averaging
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_num_runs_var).grid(row=10, column=1, padx=5,
                                                                                          pady=2, sticky="ew")

        self.run_avg_button = ttk.Button(self.avg_run_config_frame, text="Run Average Prediction",
                                         command=self._run_average_prediction_thread)
        self.run_avg_button.grid(row=11, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        self.avg_run_status_label = ttk.Label(self.avg_run_config_frame, text="Ready.")
        self.avg_run_status_label.grid(row=12, column=0, columnspan=3, padx=5, pady=2, sticky="w")

        # --- Averaged Metrics Frame (Average Single Run Tab) ---
        self.avg_run_metrics_frame = ttk.LabelFrame(self.left_panel_avg_run, text="Averaged Performance Metrics")
        self.avg_run_metrics_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.avg_run_metrics_frame.grid_columnconfigure(1, weight=1)

        self.avg_run_metric_labels = {}
        # Metrics are the same as single prediction, but will be averaged
        for i, metric_name in enumerate(["Avg MSE", "Avg R2", "Avg MAPE", "Avg Accuracy (0-1)"]):
            ttk.Label(self.avg_run_metrics_frame, text=f"{metric_name}:").grid(row=i, column=0, padx=5, pady=2,
                                                                               sticky="w")
            self.avg_run_metric_labels[metric_name] = ttk.Label(self.avg_run_metrics_frame, text="N/A")
            self.avg_run_metric_labels[metric_name].grid(row=i, column=1, padx=5, pady=2, sticky="ew")

        # --- Average Run Plot (Now functional) ---
        self.avg_run_plot_frame = ttk.LabelFrame(self.average_run_tab, text="Example Prediction Plot (Last Run)")
        # Make the plot span both rows (config and metrics) on the right side
        self.avg_run_plot_frame.grid(row=0, column=1, rowspan=1, padx=5, pady=5, sticky="nsew")
        self.avg_run_plot_frame.grid_columnconfigure(0, weight=1)
        self.avg_run_plot_frame.grid_rowconfigure(0, weight=1)

        self.fig_avg, self.ax_avg = plt.subplots(figsize=(10, 6))
        self.canvas_avg = FigureCanvasTkAgg(self.fig_avg, master=self.avg_run_plot_frame)
        self.canvas_avg_widget = self.canvas_avg.get_tk_widget()
        self.canvas_avg_widget.grid(row=0, column=0, sticky="nsew")

        self.toolbar_avg_frame = ttk.Frame(self.avg_run_plot_frame)
        self.toolbar_avg_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar_avg = NavigationToolbar2Tk(self.canvas_avg, self.toolbar_avg_frame)
        self.toolbar_avg.update()

        # Annotation for average plot (optional, but good for consistency)
        self.annot_avg = self.ax_avg.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                              bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                                              arrowprops=dict(arrowstyle="->"))
        self.annot_avg.set_visible(False)
        self.canvas_avg.mpl_connect("motion_notify_event", self._hover_avg_plot)
        self._clear_avg_run_plot()

        # --- Tab 3: Experiment ---
        self.experiment_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.experiment_tab, text="Experiment")
        # Configure columns for the split layout
        self.experiment_tab.grid_columnconfigure(0, weight=1)  # Left config panel
        self.experiment_tab.grid_columnconfigure(1, weight=1)  # Right config panel
        self.experiment_tab.grid_rowconfigure(0, weight=0)  # Config frames row (takes minimal height)
        self.experiment_tab.grid_rowconfigure(1, weight=1)  # Results frame row (expands vertically)

        # --- Experiment Configuration (Left Half) ---
        self.experiment_config_left_frame = ttk.LabelFrame(self.experiment_tab, text="General Experiment Parameters")
        self.experiment_config_left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.experiment_config_left_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.experiment_config_left_frame, text="Look Backs (comma-sep):").grid(row=0, column=0, padx=5,
                                                                                          pady=2, sticky="w")
        self.experiment_lookbacks_var = tk.StringVar(value="30,60,90")
        ttk.Entry(self.experiment_config_left_frame, textvariable=self.experiment_lookbacks_var).grid(row=0, column=1,
                                                                                                      padx=5, pady=2,
                                                                                                      sticky="ew")

        ttk.Label(self.experiment_config_left_frame, text="Number of Runs per Config:").grid(row=1, column=0, padx=5,
                                                                                             pady=2, sticky="w")
        self.num_experiment_runs_var = tk.IntVar(value=3)  # Reduced default runs to keep it manageable
        ttk.Entry(self.experiment_config_left_frame, textvariable=self.num_experiment_runs_var).grid(row=1, column=1,
                                                                                                     padx=5, pady=2,
                                                                                                     sticky="ew")

        self.run_experiment_button = ttk.Button(self.experiment_config_left_frame, text="Run Experiment",
                                                command=self._run_experiment_thread)
        self.run_experiment_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        self.experiment_status_label = ttk.Label(self.experiment_config_left_frame, text="Ready.")
        self.experiment_status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky="w")

        # --- Experiment Configuration (Right Half) ---
        self.experiment_config_right_frame = ttk.LabelFrame(self.experiment_tab, text="Hyperparameters to Tune")
        self.experiment_config_right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.experiment_config_right_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.experiment_config_right_frame, text="Dropout Rates (comma-sep):").grid(row=0, column=0, padx=5,
                                                                                              pady=2, sticky="w")
        self.experiment_dropout_rates_var = tk.StringVar(value="0.0,0.2,0.5")
        ttk.Entry(self.experiment_config_right_frame, textvariable=self.experiment_dropout_rates_var).grid(row=0,
                                                                                                           column=1,
                                                                                                           padx=5,
                                                                                                           pady=2,
                                                                                                           sticky="ew")

        ttk.Label(self.experiment_config_right_frame, text="LSTM Units (comma-sep):").grid(row=1, column=0, padx=5,
                                                                                           pady=2, sticky="w")
        self.experiment_lstm_units_var = tk.StringVar(value="50,100")
        ttk.Entry(self.experiment_config_right_frame, textvariable=self.experiment_lstm_units_var).grid(row=1, column=1,
                                                                                                        padx=5, pady=2,
                                                                                                        sticky="ew")

        ttk.Label(self.experiment_config_right_frame, text="LSTM Layers (comma-sep):").grid(row=2, column=0, padx=5,
                                                                                            pady=2, sticky="w")
        self.experiment_lstm_layers_var = tk.StringVar(value="1,2")
        ttk.Entry(self.experiment_config_right_frame, textvariable=self.experiment_lstm_layers_var).grid(row=2,
                                                                                                         column=1,
                                                                                                         padx=5, pady=2,
                                                                                                         sticky="ew")

        ttk.Label(self.experiment_config_right_frame, text="Learning Rates (comma-sep):").grid(row=3, column=0, padx=5,
                                                                                               pady=2, sticky="w")
        self.experiment_learning_rates_var = tk.StringVar(value="0.001,0.0001")
        ttk.Entry(self.experiment_config_right_frame, textvariable=self.experiment_learning_rates_var).grid(row=3,
                                                                                                            column=1,
                                                                                                            padx=5,
                                                                                                            pady=2,
                                                                                                            sticky="ew")

        # --- Experiment Results Display (Experiment Tab) ---
        self.experiment_results_display_frame = ttk.LabelFrame(self.experiment_tab,
                                                               text="Experiment Results (Average Metrics)")
        # Position this frame in the second row, spanning both columns
        self.experiment_results_display_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.experiment_results_display_frame.grid_columnconfigure(0, weight=1)
        self.experiment_results_display_frame.grid_rowconfigure(0, weight=1)  # Treeview row
        self.experiment_results_display_frame.grid_rowconfigure(1, weight=0)  # Scrollbar for treeview
        self.experiment_results_display_frame.grid_rowconfigure(2, weight=1)  # Plot row

        # Updated Treeview columns for all new hyperparameters
        exp_columns = ("Look Back", "Dropout", "Units", "Layers", "LR", "Avg MSE", "Avg R2", "Avg MAPE", "Avg Accuracy")
        self.exp_results_tree = ttk.Treeview(self.experiment_results_display_frame, columns=exp_columns,
                                             show="headings")

        for col in exp_columns:
            self.exp_results_tree.heading(col, text=col)
            self.exp_results_tree.column(col, width=80, anchor="center")  # Adjusted width for more columns

        self.exp_results_tree.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        tree_scrollbar_y = ttk.Scrollbar(self.experiment_results_display_frame, orient="vertical",
                                         command=self.exp_results_tree.yview)
        tree_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.exp_results_tree.configure(yscrollcommand=tree_scrollbar_y.set)

        tree_scrollbar_x = ttk.Scrollbar(self.experiment_results_display_frame, orient="horizontal",
                                         command=self.exp_results_tree.xview)
        tree_scrollbar_x.grid(row=1, column=0, sticky="ew")  # Placed in row 1 for Treeview's horizontal scrollbar
        self.exp_results_tree.configure(xscrollcommand=tree_scrollbar_x.set)

        # Plot for Experiment Tab
        self.fig_exp, self.ax_exp = plt.subplots(figsize=(10, 4))
        self.canvas_exp = FigureCanvasTkAgg(self.fig_exp, master=self.experiment_results_display_frame)
        self.canvas_exp_widget = self.canvas_exp.get_tk_widget()
        self.canvas_exp_widget.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")  # Placed in row 2
        self._clear_exp_plot()

        # --- Tab 4: Absolute Testing ---
        self.absolute_testing_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.absolute_testing_tab, text="Absolute Testing")
        self.absolute_testing_tab.grid_columnconfigure(0, weight=1)
        self.absolute_testing_tab.grid_columnconfigure(1, weight=2)
        self.absolute_testing_tab.grid_rowconfigure(0, weight=1)

        # Left side of Absolute Testing Tab (Config + Metrics)
        self.left_panel_absolute = ttk.Frame(self.absolute_testing_tab)
        self.left_panel_absolute.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_panel_absolute.grid_rowconfigure(0, weight=0)  # Config
        self.left_panel_absolute.grid_rowconfigure(1, weight=0)  # Description
        self.left_panel_absolute.grid_rowconfigure(2, weight=1)  # Metrics

        # --- Configuration LabelFrame (Absolute Testing Tab) ---
        self.abs_config_frame = ttk.LabelFrame(self.left_panel_absolute, text="Absolute Test Configuration")
        self.abs_config_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.abs_config_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.abs_config_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.abs_ticker_var = tk.StringVar(value="^GSPC")
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_ticker_var).grid(row=0, column=1, padx=5, pady=2,
                                                                                sticky="ew")

        # --- NEW: Train Data Start Date for Absolute Testing ---
        ttk.Label(self.abs_config_frame, text="Train Data Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5,
                                                                                          pady=2, sticky="w")
        self.abs_train_start_date_var = tk.StringVar(value=ten_years_ago)  # Default to 10 years ago
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_train_start_date_var).grid(row=1, column=1, padx=5,
                                                                                          pady=2, sticky="ew")
        # --- END NEW ---

        ttk.Label(self.abs_config_frame, text="Train Data End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2,
                                                                                        sticky="w")
        self.abs_train_end_date_var = tk.StringVar(
            value=(today - datetime.timedelta(days=30)).strftime('%Y-%m-%d'))  # Default to 1 month ago
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_train_end_date_var).grid(row=2, column=1, padx=5, pady=2,
                                                                                        sticky="ew")

        ttk.Label(self.abs_config_frame, text="Prediction Start Date (YYYY-MM-DD):").grid(row=3, column=0, padx=5,
                                                                                          pady=2, sticky="w")
        self.abs_pred_start_date_var = tk.StringVar(
            value=(today - datetime.timedelta(days=29)).strftime('%Y-%m-%d'))  # Default to day after train_end
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_pred_start_date_var).grid(row=3, column=1, padx=5,
                                                                                         pady=2, sticky="ew")

        ttk.Label(self.abs_config_frame, text="Prediction End Date (YYYY-MM-DD):").grid(row=4, column=0, padx=5, pady=2,
                                                                                        sticky="w")
        self.abs_pred_end_date_var = tk.StringVar(value=yesterday)  # Default to yesterday
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_pred_end_date_var).grid(row=4, column=1, padx=5, pady=2,
                                                                                       sticky="ew")

        ttk.Label(self.abs_config_frame, text="Look Back Window (days):").grid(row=5, column=0, padx=5, pady=2,
                                                                               sticky="w")
        self.abs_look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_look_back_var).grid(row=5, column=1, padx=5, pady=2,
                                                                                   sticky="ew")

        ttk.Label(self.abs_config_frame, text="Model Architecture:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.abs_model_var = tk.StringVar(value="Conv1D + LSTM")
        self.abs_model_dropdown = ttk.Combobox(self.abs_config_frame, textvariable=self.abs_model_var,
                                               values=list(model_builders.keys()), state="readonly")
        self.abs_model_dropdown.grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        # Hyperparameters for absolute testing tab (re-using single run params as sensible defaults)
        ttk.Label(self.abs_config_frame, text="LSTM Units:").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.abs_lstm_units_var = tk.IntVar(value=50)
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_lstm_units_var).grid(row=7, column=1, padx=5, pady=2,
                                                                                    sticky="ew")

        ttk.Label(self.abs_config_frame, text="LSTM Layers:").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.abs_lstm_layers_var = tk.IntVar(value=1)
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_lstm_layers_var).grid(row=8, column=1, padx=5, pady=2,
                                                                                     sticky="ew")

        ttk.Label(self.abs_config_frame, text="Dropout Rate:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        self.abs_dropout_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_dropout_rate_var).grid(row=9, column=1, padx=5, pady=2,
                                                                                      sticky="ew")

        ttk.Label(self.abs_config_frame, text="Learning Rate:").grid(row=10, column=0, padx=5, pady=2, sticky="w")
        self.abs_learning_rate_var = tk.DoubleVar(value=0.001)
        ttk.Entry(self.abs_config_frame, textvariable=self.abs_learning_rate_var).grid(row=10, column=1, padx=5, pady=2,
                                                                                       sticky="ew")

        self.run_abs_button = ttk.Button(self.abs_config_frame, text="Run Absolute Test",
                                         command=self._run_absolute_testing_thread)
        self.run_abs_button.grid(row=11, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        self.abs_status_label = ttk.Label(self.abs_config_frame, text="Ready.")
        self.abs_status_label.grid(row=12, column=0, columnspan=2, padx=5, pady=2, sticky="w")

        # --- Absolute Testing Description ---
        self.abs_description_frame = ttk.LabelFrame(self.left_panel_absolute, text="About Absolute Testing")
        self.abs_description_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        description_text = (
            "In Absolute Testing, the model is trained *only* on data up to the 'Train Data End Date'. "
            "It is then used to predict prices for the 'Prediction Start Date' to 'Prediction End Date', "
            "a period for which the model has *never seen any data*. "
            "This simulates real-world forward-looking prediction and helps evaluate generalization ability."
        )
        ttk.Label(self.abs_description_frame, text=description_text, wraplength=350, justify=tk.LEFT).pack(padx=5,
                                                                                                           pady=5)

        # --- Metrics Frame (Absolute Testing Tab) ---
        self.abs_metrics_frame = ttk.LabelFrame(self.left_panel_absolute, text="Prediction Metrics")
        self.abs_metrics_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.abs_metrics_frame.grid_columnconfigure(1, weight=1)

        self.abs_metric_labels = {}
        abs_metrics_order = ["MSE", "R2", "MAPE", "Accuracy (0-1)", "Prediction Period", "Num Predictions"]
        for i, metric_name in enumerate(abs_metrics_order):
            ttk.Label(self.abs_metrics_frame, text=f"{metric_name}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.abs_metric_labels[metric_name] = ttk.Label(self.abs_metrics_frame, text="N/A")
            self.abs_metric_labels[metric_name].grid(row=i, column=1, padx=5, pady=2, sticky="ew")

        # --- Absolute Testing Plot ---
        self.abs_plot_frame = ttk.LabelFrame(self.absolute_testing_tab, text="Absolute Prediction Plot")
        self.abs_plot_frame.grid(row=0, column=1, rowspan=1, padx=5, pady=5, sticky="nsew")
        self.abs_plot_frame.grid_columnconfigure(0, weight=1)
        self.abs_plot_frame.grid_rowconfigure(0, weight=1)

        self.fig_abs, self.ax_abs = plt.subplots(figsize=(10, 6))
        self.canvas_abs = FigureCanvasTkAgg(self.fig_abs, master=self.abs_plot_frame)
        self.canvas_abs_widget = self.canvas_abs.get_tk_widget()
        self.canvas_abs_widget.grid(row=0, column=0, sticky="nsew")

        self.toolbar_abs_frame = ttk.Frame(self.abs_plot_frame)
        self.toolbar_abs_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar_abs = NavigationToolbar2Tk(self.canvas_abs, self.toolbar_abs_frame)
        self.toolbar_abs.update()

        self.annot_abs = self.ax_abs.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                              bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                                              arrowprops=dict(arrowstyle="->"))
        self.annot_abs.set_visible(False)
        self.canvas_abs.mpl_connect("motion_notify_event", self._hover_abs_plot)
        self._clear_abs_plot()

    def _update_train_ratio_label(self, val):
        self.train_ratio_label.config(text=f"{float(val):.2f}")

    def _update_avg_run_train_ratio_label(self, val):
        self.avg_run_train_ratio_label.config(text=f"{float(val):.2f}")

    def _clear_main_plot(self):
        self.ax_main.clear()
        self.ax_main.set_title('Single Prediction Plot')
        self.ax_main.set_xlabel('Date')
        self.ax_main.set_ylabel('Stock Price')
        self.ax_main.grid(True)
        self.canvas_main.draw()
        self.annot_main.set_visible(False)

    def _clear_avg_run_plot(self):
        self.ax_avg.clear()
        self.ax_avg.set_title('Average Run: Example Prediction Plot')
        self.ax_avg.set_xlabel('Date')
        self.ax_avg.set_ylabel('Stock Price')
        self.ax_avg.grid(True)
        self.canvas_avg.draw()
        self.annot_avg.set_visible(False)

    def _clear_exp_plot(self):
        self.ax_exp.clear()
        self.ax_exp.set_title('Experiment Summary: Average Metrics by Look Back (Initial Plot)')  # Updated title
        self.ax_exp.set_xlabel('Look Back Window')
        self.ax_exp.set_ylabel('Metric Value')
        self.ax_exp.grid(True)
        self.canvas_exp.draw()

    def _clear_abs_plot(self):
        self.ax_abs.clear()
        self.ax_abs.set_title('Absolute Prediction Plot')
        self.ax_abs.set_xlabel('Date')
        self.ax_abs.set_ylabel('Stock Price')
        self.ax_abs.grid(True)
        self.canvas_abs.draw()
        self.annot_abs.set_visible(False)

    def _hover_main_plot(self, event):
        if event.inaxes == self.ax_main:
            for line in self.ax_main.lines:
                contains, attr = line.contains(event)
                if contains:
                    ind = attr['ind'][0]
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()

                    date_val = plt.num2date(xdata[ind])
                    date_str = date_val.strftime('%Y-%m-%d')
                    price_str = f"{ydata[ind]:.2f}"

                    self.annot_main.xy = (xdata[ind], ydata[ind])
                    text = f"Date: {date_str}\nPrice: {price_str}"
                    self.annot_main.set_text(text)
                    self.annot_main.set_visible(True)
                    self.fig_main.canvas.draw_idle()
                    return
        self.annot_main.set_visible(False)
        self.fig_main.canvas.draw_idle()

    def _hover_avg_plot(self, event):
        if event.inaxes == self.ax_avg:
            for line in self.ax_avg.lines:
                contains, attr = line.contains(event)
                if contains:
                    ind = attr['ind'][0]
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()

                    date_val = plt.num2date(xdata[ind])
                    date_str = date_val.strftime('%Y-%m-%d')
                    price_str = f"{ydata[ind]:.2f}"

                    self.annot_avg.xy = (xdata[ind], ydata[ind])
                    text = f"Date: {date_str}\nPrice: {price_str}"
                    self.annot_avg.set_text(text)
                    self.annot_avg.set_visible(True)
                    self.fig_avg.canvas.draw_idle()
                    return
        self.annot_avg.set_visible(False)
        self.fig_avg.canvas.draw_idle()

    def _hover_abs_plot(self, event):
        if event.inaxes == self.ax_abs:
            for line in self.ax_abs.lines:
                contains, attr = line.contains(event)
                if contains:
                    ind = attr['ind'][0]
                    xdata = line.get_xdata()
                    ydata = line.get_ydata()

                    date_val = plt.num2date(xdata[ind])
                    date_str = date_val.strftime('%Y-%m-%d')
                    price_str = f"{ydata[ind]:.2f}"

                    self.annot_abs.xy = (xdata[ind], ydata[ind])
                    text = f"Date: {date_str}\nPrice: {price_str}"
                    self.annot_abs.set_text(text)
                    self.annot_abs.set_visible(True)
                    self.fig_abs.canvas.draw_idle()
                    return
        self.annot_abs.set_visible(False)
        self.fig_abs.canvas.draw_idle()

    # --- Single Prediction Logic ---
    def _run_single_prediction_thread(self):
        # Disable all run buttons across tabs to prevent concurrent operations
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)  # Disable new button
        self.status_label.config(text="Running single prediction... Please wait.")
        self.avg_run_status_label.config(text="")
        self.experiment_status_label.config(text="")
        self.abs_status_label.config(text="")  # Clear new status label
        self.master.update_idletasks()

        thread = threading.Thread(target=self._perform_single_prediction_task)
        thread.start()

    def _perform_single_prediction_task(self):
        try:
            metrics, plot_data = self._run_core_prediction_logic(
                ticker=self.ticker_var.get(),
                train_start_date_str=self.start_date_var.get(),
                train_end_date_str=self.end_date_var.get(),
                pred_start_date_str=None,  # Not used in single prediction, handled by train/test split
                pred_end_date_str=None,  # Not used in single prediction, handled by train/test split
                look_back=self.look_back_var.get(),
                train_test_split_ratio=self.train_ratio_var.get(),
                selected_model_name=self.model_var.get(),
                lstm_units=self.single_lstm_units_var.get(),  # Pass fixed single run params
                num_lstm_layers=self.single_lstm_layers_var.get(),
                dropout_rate=self.single_dropout_rate_var.get(),
                learning_rate=self.single_learning_rate_var.get(),
                is_absolute_test=False  # Indicate this is not an absolute test
            )

            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    if "MAPE" in metric_name:
                        self.metric_labels[metric_name].config(text=f"{value:.2f}%")
                    elif "Accuracy" in metric_name:
                        self.metric_labels[metric_name].config(text=f"{value:.4f}")
                    else:
                        self.metric_labels[metric_name].config(text=f"{value:.4f}")
                else:
                    self.metric_labels[metric_name].config(text=str(value))

            self.ax_main.clear()
            self.ax_main.plot(plot_data['test_dates'], plot_data['actual_values'], label='Actual Values', color='black',
                              linewidth=2)
            self.ax_main.plot(plot_data['test_dates'], plot_data['test_predict'],
                              label=f'{self.model_var.get()} Predicted', color='purple', linestyle='--')
            self.ax_main.set_title(f'{self.ticker_var.get()} Stock Price Prediction: Actual vs. Predicted Values')
            self.ax_main.set_xlabel('Date')
            self.ax_main.set_ylabel('Stock Price')
            self.ax_main.legend()
            self.ax_main.grid(True)
            self.fig_main.tight_layout()
            self.canvas_main.draw()

            self.status_label.config(text="Single prediction complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Error occurred during single prediction.")
            self._clear_main_plot()
        finally:
            # Re-enable all run buttons
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)  # Enable new button

    # --- Core Prediction Logic (returns data, doesn't update GUI directly) ---
    # Updated to accept new hyperparameters and absolute test flag
    def _run_core_prediction_logic(self, ticker, train_start_date_str, train_end_date_str,
                                   pred_start_date_str, pred_end_date_str,  # New parameters for absolute testing
                                   look_back, train_test_split_ratio,
                                   selected_model_name, lstm_units, num_lstm_layers, dropout_rate, learning_rate,
                                   is_absolute_test=False):  # New flag
        # Input validation
        if not ticker or not train_start_date_str or not train_end_date_str:
            raise ValueError("Ticker and training dates are required.")
        if is_absolute_test and (not pred_start_date_str or not pred_end_date_str):
            raise ValueError("Prediction dates are required for Absolute Testing.")

        try:
            train_start_date_dt = datetime.datetime.strptime(train_start_date_str, '%Y-%m-%d')
            train_end_date_dt = datetime.datetime.strptime(train_end_date_str, '%Y-%m-%d')
            if train_start_date_dt >= train_end_date_dt:
                raise ValueError("Training start date must be before training end date.")

            if is_absolute_test:
                pred_start_date_dt = datetime.datetime.strptime(pred_start_date_str, '%Y-%m-%d')
                pred_end_date_dt = datetime.datetime.strptime(pred_end_date_str, '%Y-%m-%d')
                if pred_start_date_dt >= pred_end_date_dt:
                    raise ValueError("Prediction start date must be before prediction end date.")
                if train_end_date_dt >= pred_start_date_dt:
                    raise ValueError("Training data must end before the prediction period begins for Absolute Testing.")
        except ValueError as ve:
            raise ValueError(f"Invalid date format or range: {ve}. Use YYYY-MM-DD.")

        if look_back <= 0:
            raise ValueError("Look back window must be a positive integer.")
        if not (0.1 <= train_test_split_ratio <= 0.99) and not is_absolute_test:  # Ratio not used in absolute test
            raise ValueError("Train/Test ratio must be between 0.1 and 0.99.")
        if lstm_units <= 0 or num_lstm_layers <= 0:
            raise ValueError("LSTM Units and Layers must be positive integers.")
        if not (0 <= dropout_rate <= 1):
            raise ValueError("Dropout Rate must be between 0 and 1.")
        if learning_rate <= 0:
            raise ValueError("Learning Rate must be positive.")

        # --- Data for Training (and potentially in-sample testing for non-absolute mode) ---
        train_data_raw = get_stock_data_live(ticker, train_start_date_dt.strftime('%Y-%m-%d'),
                                             train_end_date_dt.strftime('%Y-%m-%d'))
        if train_data_raw.empty or len(train_data_raw) < look_back + 1:
            raise ValueError(
                f"Not enough training data for ticker {ticker} from {train_start_date_str} to {train_end_date_str} with look-back {look_back}.")

        train_data_raw = train_data_raw.sort_index()
        features = ['Close']
        train_data_for_scaling = train_data_raw[features].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data_for_scaling)

        X_train_full, Y_train_full = create_sequences(scaled_train_data, look_back)
        if len(X_train_full) == 0:
            raise ValueError(
                f"Not enough data to create training sequences with look-back {look_back}. Try reducing look-back or extending training data range.")

        selected_model_builder = model_builders[selected_model_name]
        model = selected_model_builder((look_back, 1), lstm_units, num_lstm_layers, dropout_rate, learning_rate)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Reshape for LSTM
        X_train_full = np.reshape(X_train_full, (X_train_full.shape[0], X_train_full.shape[1], 1))

        if not is_absolute_test:  # Normal train/test split within the provided date range
            train_size = int(len(X_train_full) * train_test_split_ratio)
            if train_size < 1 or (len(X_train_full) - train_size) < 1:
                raise ValueError(
                    f"Invalid train/test split for look-back {look_back}. Ensure at least one sample for training and one for testing. Adjust ratio or data range.")

            X_train, X_test = X_train_full[0:train_size], X_train_full[train_size:len(X_train_full)]
            Y_train, Y_test = Y_train_full[0:train_size], Y_train_full[train_size:len(Y_train_full)]

            model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0,
                      callbacks=[early_stopping])

            test_predict_scaled = model.predict(X_test)
            # Reconstruct dummy arrays for inverse transform
            dummy_predictions = np.zeros((len(test_predict_scaled), len(features)))
            dummy_predictions[:, 0] = test_predict_scaled.flatten()
            test_predict = scaler.inverse_transform(dummy_predictions)[:, 0]

            dummy_Y_test = np.zeros((len(Y_test), len(features)))
            dummy_Y_test[:, 0] = Y_test.flatten()
            actual_values = scaler.inverse_transform(dummy_Y_test)[:, 0]

            # Adjust test dates to match the actual_values
            test_dates_original_index = train_data_raw.index[train_size + look_back:]
            test_dates = test_dates_original_index[:len(actual_values)]  # Ensure length matches

        else:  # Absolute Testing: Train on all available train_data_raw, predict on future data
            model.fit(X_train_full, Y_train_full, epochs=100, batch_size=32, validation_split=0.1, verbose=0,
                      callbacks=[early_stopping])

            # --- Data for Prediction (future, unseen data) ---
            prediction_data_raw = get_stock_data_live(ticker, pred_start_date_dt.strftime('%Y-%m-%d'),
                                                      pred_end_date_dt.strftime('%Y-%m-%d'))
            if prediction_data_raw.empty:
                raise ValueError(
                    f"No data available for the prediction period for ticker {ticker} from {pred_start_date_str} to {pred_end_date_str}.")

            prediction_data_raw = prediction_data_raw.sort_index()
            actual_values = prediction_data_raw['Close'].values
            test_dates = prediction_data_raw.index

            # For prediction, we need to feed a "rolling window" of data that the model has seen
            # The last 'look_back' days of the training data are needed to start the prediction
            # Then, for subsequent predictions, we use the actual previous values.

            # Get the last 'look_back' days from the training data for the first prediction
            last_train_data = train_data_raw['Close'].tail(look_back).values

            # Scale this initial segment using the *same scaler* fitted on training data
            current_input = scaler.transform(last_train_data.reshape(-1, 1))
            current_input = np.reshape(current_input, (1, look_back, 1))

            test_predict_scaled = []

            for i in range(len(actual_values)):
                # Predict the next day's price
                predicted_price_scaled = model.predict(current_input, verbose=0)[0, 0]
                test_predict_scaled.append(predicted_price_scaled)

                # Update current_input for the next prediction
                # Use the actual value from the prediction period for the next input,
                # to maintain realism and prevent error accumulation from predicted values.
                if i < len(actual_values) - 1:
                    next_actual_scaled = scaler.transform(np.array(actual_values[i]).reshape(-1, 1))[0, 0]
                    # Shift the window, remove the oldest, add the newest actual value
                    current_input = np.append(current_input[:, 1:, :], next_actual_scaled).reshape(1, look_back, 1)

            # Inverse transform the predictions
            dummy_predictions = np.zeros((len(test_predict_scaled), len(features)))
            dummy_predictions[:, 0] = test_predict_scaled
            test_predict = scaler.inverse_transform(dummy_predictions)[:, 0]

        # Common metric calculation
        if len(actual_values) == 0:
            raise ValueError(
                "No actual values to compare against for metrics. Prediction period might be empty or invalid.")

        mse = mean_squared_error(actual_values, test_predict)
        # Handle division by zero for MAPE gracefully
        mape = np.mean(np.abs((actual_values - test_predict) / np.where(actual_values == 0, 1e-9, actual_values))) * 100
        r2 = r2_score(actual_values, test_predict)
        accuracy_score_0_1 = max(0, 1 - (mape / 100))

        metrics_results = {
            "MSE": mse,
            "R2": r2,
            "MAPE": mape,
            "Accuracy (0-1)": accuracy_score_0_1,
            "Prediction Period": f"{test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}",
            "Num Predictions": len(test_dates)
        }

        plot_data = {
            'test_dates': test_dates,
            'actual_values': actual_values,
            'test_predict': test_predict,
            'model_name': selected_model_name
        }

        return metrics_results, plot_data

    # --- Average Single Run Logic ---
    def _run_average_prediction_thread(self):
        # Disable all run buttons across tabs
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)  # Disable new button
        self.avg_run_status_label.config(text="Running average prediction... Please wait.")
        self.status_label.config(text="")
        self.experiment_status_label.config(text="")
        self.abs_status_label.config(text="")  # Clear new status label
        self.master.update_idletasks()

        thread = threading.Thread(target=self._perform_average_prediction_task)
        thread.start()

    def _perform_average_prediction_task(self):
        try:
            num_runs = self.avg_run_num_runs_var.get()
            if num_runs <= 0:
                raise ValueError("Number of runs for average prediction must be a positive integer.")

            all_mses = []
            all_r2s = []
            all_mapes = []
            all_accuracies = []
            last_plot_data = None  # Store plot data for the last successful run

            self._clear_avg_run_plot()  # Clear plot at the beginning

            for run_idx in range(num_runs):
                self.avg_run_status_label.config(text=f"Running average prediction: Run {run_idx + 1}/{num_runs}...")
                self.master.update_idletasks()

                try:
                    # Note: Average run uses train_start and train_end for its data fetch,
                    # and then internally splits.
                    metrics, plot_data_current = self._run_core_prediction_logic(
                        ticker=self.avg_run_ticker_var.get(),
                        train_start_date_str=self.avg_run_start_date_var.get(),
                        train_end_date_str=self.avg_run_end_date_var.get(),
                        pred_start_date_str=None,
                        pred_end_date_str=None,
                        look_back=self.avg_run_look_back_var.get(),
                        train_test_split_ratio=self.avg_run_train_ratio_var.get(),
                        selected_model_name=self.avg_run_model_var.get(),
                        lstm_units=self.avg_run_lstm_units_var.get(),
                        num_lstm_layers=self.avg_run_lstm_layers_var.get(),
                        dropout_rate=self.avg_run_dropout_rate_var.get(),
                        learning_rate=self.avg_run_learning_rate_var.get(),
                        is_absolute_test=False
                    )
                    all_mses.append(metrics['MSE'])
                    all_r2s.append(metrics['R2'])
                    all_mapes.append(metrics['MAPE'])
                    all_accuracies.append(metrics['Accuracy (0-1)'])
                    last_plot_data = plot_data_current  # Keep plot data from the last successful run

                except Exception as e:
                    print(f"Error during average run {run_idx + 1}: {e}")
                    messagebox.showwarning("Average Run Error",
                                           f"Failed during run {run_idx + 1}: {e}. Skipping this run.")
                    # Continue to next run if one fails, but don't include failed metrics

            if all_mses:  # Ensure we have at least one successful run
                avg_mse = np.mean(all_mses)
                avg_r2 = np.mean(all_r2s)
                avg_mape = np.mean(all_mapes)
                avg_accuracy = np.mean(all_accuracies)

                self.avg_run_metric_labels["Avg MSE"].config(text=f"{avg_mse:.4f}")
                self.avg_run_metric_labels["Avg R2"].config(text=f"{avg_r2:.4f}")
                self.avg_run_metric_labels["Avg MAPE"].config(text=f"{avg_mape:.2f}%")
                self.avg_run_metric_labels["Avg Accuracy (0-1)"].config(text=f"{avg_accuracy:.4f}")
                self.avg_run_status_label.config(
                    text=f"Average prediction complete over {len(all_mses)} successful runs!")

                # Plot the last successful run's data
                if last_plot_data:
                    self.ax_avg.clear()
                    self.ax_avg.plot(last_plot_data['test_dates'], last_plot_data['actual_values'],
                                     label='Actual Values', color='black', linewidth=2)
                    self.ax_avg.plot(last_plot_data['test_dates'], last_plot_data['test_predict'],
                                     label=f'{last_plot_data["model_name"]} Predicted', color='purple', linestyle='--')
                    self.ax_avg.set_title(
                        f'{self.avg_run_ticker_var.get()} Stock Price: Example Run (Last of {len(all_mses)})')
                    self.ax_avg.set_xlabel('Date')
                    self.ax_avg.set_ylabel('Stock Price')
                    self.ax_avg.legend()
                    self.ax_avg.grid(True)
                    self.fig_avg.tight_layout()
                    self.canvas_avg.draw()
            else:
                self.avg_run_status_label.config(text="No successful runs to average. Check configurations.")
                for label in self.avg_run_metric_labels.values():
                    label.config(text="N/A")
                self._clear_avg_run_plot()  # Ensure plot is cleared if no successful runs


        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.avg_run_status_label.config(text="Error occurred during average prediction.")
            self._clear_avg_run_plot()
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)  # Enable new button

    # --- Experiment Logic ---
    def _run_experiment_thread(self):
        # Disable all run buttons across tabs
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)  # Disable new button
        self.experiment_status_label.config(text="Running experiment... Please wait. This may take a while.")
        self.status_label.config(text="")
        self.avg_run_status_label.config(text="")
        self.abs_status_label.config(text="")  # Clear new status label
        self.master.update_idletasks()

        thread = threading.Thread(target=self._perform_experiment_task)
        thread.start()

    def _perform_experiment_task(self):
        try:
            # Parse all hyperparameter lists
            look_backs = [int(x.strip()) for x in self.experiment_lookbacks_var.get().split(',') if x.strip()]
            dropout_rates = [float(x.strip()) for x in self.experiment_dropout_rates_var.get().split(',') if x.strip()]
            lstm_units_list = [int(x.strip()) for x in self.experiment_lstm_units_var.get().split(',') if x.strip()]
            lstm_layers_list = [int(x.strip()) for x in self.experiment_lstm_layers_var.get().split(',') if x.strip()]
            learning_rates = [float(x.strip()) for x in self.experiment_learning_rates_var.get().split(',') if
                              x.strip()]
            num_runs_per_config = self.num_experiment_runs_var.get()

            # Basic validation
            if not look_backs or not dropout_rates or not lstm_units_list or not lstm_layers_list or not learning_rates or num_runs_per_config <= 0:
                raise ValueError("All experiment fields must have valid, non-empty, positive values.")

            for lb in look_backs:
                if lb <= 0: raise ValueError("Look Backs must be positive integers.")
            for dr in dropout_rates:
                if not (0 <= dr <= 1): raise ValueError("Dropout Rates must be between 0 and 1.")
            for units in lstm_units_list:
                if units <= 0: raise ValueError("LSTM Units must be positive integers.")
            for layers in lstm_layers_list:
                if layers <= 0: raise ValueError("LSTM Layers must be positive integers.")
            for lr in learning_rates:
                if lr <= 0: raise ValueError("Learning Rates must be positive.")

            # Calculate total runs and warn user
            total_combinations = len(look_backs) * len(dropout_rates) * len(lstm_units_list) * len(
                lstm_layers_list) * len(learning_rates)
            total_actual_runs = total_combinations * num_runs_per_config

            if total_actual_runs > 50:  # Arbitrary threshold, adjust as needed
                if not messagebox.askyesno("Warning: Long Experiment",
                                           f"This experiment will run {total_actual_runs} models, which may take a very long time.\n"
                                           "Do you want to continue?"):
                    self.experiment_status_label.config(text="Experiment cancelled by user.")
                    return

            all_experiment_results = []  # To store detailed results for each config

            for item in self.exp_results_tree.get_children():
                self.exp_results_tree.delete(item)
            self._clear_exp_plot()

            # Using current values from single prediction tab for ticker/dates/ratio
            current_ticker = self.ticker_var.get()
            current_start_date = self.start_date_var.get()
            current_end_date = self.end_date_var.get()
            current_train_ratio = self.train_ratio_var.get()
            current_model_name = self.model_var.get()  # Model architecture is also a fixed choice per experiment

            # Iterate through all combinations of hyperparameters
            hyperparam_combinations = itertools.product(
                look_backs, dropout_rates, lstm_units_list, lstm_layers_list, learning_rates
            )

            current_config_idx = 0
            for lb, dr, units, layers, lr in hyperparam_combinations:
                current_config_idx += 1
                individual_run_metrics_for_config = []

                for run_idx in range(num_runs_per_config):
                    self.experiment_status_label.config(
                        text=f"Exp: Config {current_config_idx}/{total_combinations} (LB:{lb}, DR:{dr}, Units:{units}, Layers:{layers}, LR:{lr}) "
                             f"Run {run_idx + 1}/{num_runs_per_config}. Model: {current_model_name}"
                    )
                    self.master.update_idletasks()

                    try:
                        # For experiment, use the train_start and train_end of the single prediction tab
                        # and let the core logic handle the internal train/test split.
                        metrics, _ = self._run_core_prediction_logic(
                            ticker=current_ticker,
                            train_start_date_str=current_start_date,
                            train_end_date_str=current_end_date,
                            pred_start_date_str=None,  # Not applicable for Experiment tab's internal split
                            pred_end_date_str=None,  # Not applicable for Experiment tab's internal split
                            look_back=lb,
                            train_test_split_ratio=current_train_ratio,
                            selected_model_name=current_model_name,
                            lstm_units=units,
                            num_lstm_layers=layers,
                            dropout_rate=dr,
                            learning_rate=lr,
                            is_absolute_test=False
                        )
                        individual_run_metrics_for_config.append({
                            'MSE': metrics['MSE'],
                            'R2': metrics['R2'],
                            'MAPE': metrics['MAPE'],
                            'Accuracy (0-1)': metrics['Accuracy (0-1)']
                        })
                    except Exception as e:
                        print(f"Error during experiment run (Config: {current_config_idx}, Run: {run_idx + 1}): {e}")
                        messagebox.showwarning("Experiment Run Error",
                                               f"Failed for Config (LB:{lb}, DR:{dr}, Units:{units}, Layers:{layers}, LR:{lr}), "
                                               f"Run {run_idx + 1}: {e}. Skipping this run.")
                        # Continue to next run if one fails, but don't include failed metrics

                if individual_run_metrics_for_config:
                    avg_mse = np.mean([r['MSE'] for r in individual_run_metrics_for_config])
                    avg_r2 = np.mean([r['R2'] for r in individual_run_metrics_for_config])
                    avg_mape = np.mean([r['MAPE'] for r in individual_run_metrics_for_config])
                    avg_accuracy = np.mean([r['Accuracy (0-1)'] for r in individual_run_metrics_for_config])

                    self.exp_results_tree.insert("", "end", values=(
                        lb, dr, units, layers, lr,
                        f"{avg_mse:.4f}",
                        f"{avg_r2:.4f}",
                        f"{avg_mape:.2f}%",
                        f"{avg_accuracy:.4f}"
                    ))
                    # Store detailed result for potential plotting later if needed
                    all_experiment_results.append({
                        'Look Back': lb, 'Dropout': dr, 'Units': units, 'Layers': layers, 'LR': lr,
                        'Avg MSE': avg_mse, 'Avg R2': avg_r2, 'Avg MAPE': avg_mape, 'Avg Accuracy': avg_accuracy
                    })
                else:
                    self.exp_results_tree.insert("", "end", values=(
                        lb, dr, units, layers, lr, "N/A", "N/A", "N/A", "N/A"
                    ))

            # Re-plot the experiment summary based on the primary variable (e.g., Look Back)
            if all_experiment_results:
                self._clear_exp_plot()
                self.ax_exp.set_title("Experiment Complete. See table for detailed results.")
                self.canvas_exp.draw()

            self.experiment_status_label.config(text="Experiment complete!")

        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
            self.experiment_status_label.config(text="Error: Check input values.")
            self._clear_exp_plot()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.experiment_status_label.config(text="Error occurred during experiment.")
            self._clear_exp_plot()
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)
            self.run_abs_button.config(state=tk.NORMAL)  # Enable new button

    # --- Absolute Testing Logic ---
    def _run_absolute_testing_thread(self):
        # Disable all run buttons across tabs
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.run_abs_button.config(state=tk.DISABLED)
        self.abs_status_label.config(text="Running absolute test... Please wait.")
        self.status_label.config(text="")
        self.avg_run_status_label.config(text="")
        self.experiment_status_label.config(text="")
        self.master.update_idletasks()

        thread = threading.Thread(target=self._perform_absolute_testing_task)
        thread.start()

    def _perform_absolute_testing_task(self):
        try:
            # Clear previous plot and metrics
            self._clear_abs_plot()
            for label in self.abs_metric_labels.values():
                label.config(text="N/A")

            metrics, plot_data = self._run_core_prediction_logic(
                ticker=self.abs_ticker_var.get(),
                train_start_date_str=self.abs_train_start_date_var.get(),  # Use the new UI variable
                train_end_date_str=self.abs_train_end_date_var.get(),
                pred_start_date_str=self.abs_pred_start_date_var.get(),
                pred_end_date_str=self.abs_pred_end_date_var.get(),
                look_back=self.abs_look_back_var.get(),
                train_test_split_ratio=0.0,  # Not used in absolute test mode
                selected_model_name=self.abs_model_var.get(),
                lstm_units=self.abs_lstm_units_var.get(),
                num_lstm_layers=self.abs_lstm_layers_var.get(),
                dropout_rate=self.abs_dropout_rate_var.get(),
                learning_rate=self.abs_learning_rate_var.get(),
                is_absolute_test=True  # Crucial flag
            )

            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    if "MAPE" in metric_name:
                        self.abs_metric_labels[metric_name].config(text=f"{value:.2f}%")
                    elif "Accuracy" in metric_name:
                        self.abs_metric_labels[metric_name].config(text=f"{value:.4f}")
                    else:
                        self.abs_metric_labels[metric_name].config(text=f"{value:.4f}")
                else:
                    self.abs_metric_labels[metric_name].config(text=str(value))

            self.ax_abs.clear()
            self.ax_abs.plot(plot_data['test_dates'], plot_data['actual_values'], label='Actual Values', color='black',
                             linewidth=2)
            self.ax_abs.plot(plot_data['test_dates'], plot_data['test_predict'],
                             label=f'{plot_data["model_name"]} Predicted', color='red', linestyle='--')
            self.ax_abs.set_title(f'{self.abs_ticker_var.get()} Absolute Test: Actual vs. Predicted Values')
            self.ax_abs.set_xlabel('Date')
            self.ax_abs.set_ylabel('Stock Price')
            self.ax_abs.legend()
            self.ax_abs.grid(True)
            self.fig_abs.tight_layout()
            self.canvas_abs.draw()

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


def main():
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()