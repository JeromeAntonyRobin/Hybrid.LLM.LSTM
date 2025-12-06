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

# --- Model Definitions ---
def build_basic_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2), # Added Dropout
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_bidirectional_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(50, return_sequences=False), input_shape=input_shape),
        Dropout(0.2), # Added Dropout
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_conv1d_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(0.2), # Added Dropout
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
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
        master.geometry("1000x800")
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
        ttk.Entry(self.config_frame, textvariable=self.start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

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

        self.run_button = ttk.Button(self.config_frame, text="Run Single Prediction", command=self._run_single_prediction_thread)
        self.run_button.grid(row=6, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        self.status_label = ttk.Label(self.config_frame, text="Ready.")
        self.status_label.grid(row=7, column=0, columnspan=3, padx=5, pady=2, sticky="w")

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
        self.average_run_tab.grid_columnconfigure(0, weight=1)
        self.average_run_tab.grid_columnconfigure(1, weight=1) # For future plot if needed, currently empty
        self.average_run_tab.grid_rowconfigure(0, weight=0) # Config frame
        self.average_run_tab.grid_rowconfigure(1, weight=1) # Metrics frame

        # Left side of Average Single Run Tab (Config + Metrics)
        self.left_panel_avg_run = ttk.Frame(self.average_run_tab)
        self.left_panel_avg_run.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_panel_avg_run.grid_rowconfigure(0, weight=0)
        self.left_panel_avg_run.grid_rowconfigure(1, weight=1)

        # --- Configuration LabelFrame (Average Single Run Tab) ---
        self.avg_run_config_frame = ttk.LabelFrame(self.left_panel_avg_run, text="Average Run Configuration")
        self.avg_run_config_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.avg_run_config_frame.grid_columnconfigure(1, weight=1)

        # Re-using default values, but variables are distinct
        ttk.Label(self.avg_run_config_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_ticker_var = tk.StringVar(value="^GSPC")
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_ticker_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_start_date_var = tk.StringVar(value=ten_years_ago)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_start_date_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_end_date_var = tk.StringVar(value=yesterday)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_end_date_var).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Look Back Window (days):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_look_back_var = tk.IntVar(value=60)
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_look_back_var).grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Train/Test Ratio:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_train_ratio_var = tk.DoubleVar(value=0.9)
        self.avg_run_train_ratio_slider = ttk.Scale(self.avg_run_config_frame, from_=0.1, to=0.99, orient="horizontal",
                                            variable=self.avg_run_train_ratio_var, command=self._update_avg_run_train_ratio_label)
        self.avg_run_train_ratio_slider.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        self.avg_run_train_ratio_label = ttk.Label(self.avg_run_config_frame, text=f"{self.avg_run_train_ratio_var.get():.2f}")
        self.avg_run_train_ratio_label.grid(row=4, column=2, padx=5, pady=2, sticky="w")

        ttk.Label(self.avg_run_config_frame, text="Model Architecture:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_model_var = tk.StringVar(value="Conv1D + LSTM")
        self.avg_run_model_dropdown = ttk.Combobox(self.avg_run_config_frame, textvariable=self.avg_run_model_var,
                                          values=list(model_builders.keys()), state="readonly")
        self.avg_run_model_dropdown.grid(row=5, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.avg_run_config_frame, text="Number of Runs:").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.avg_run_num_runs_var = tk.IntVar(value=10) # Default to 10 runs for averaging
        ttk.Entry(self.avg_run_config_frame, textvariable=self.avg_run_num_runs_var).grid(row=6, column=1, padx=5, pady=2, sticky="ew")

        self.run_avg_button = ttk.Button(self.avg_run_config_frame, text="Run Average Prediction", command=self._run_average_prediction_thread)
        self.run_avg_button.grid(row=7, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

        self.avg_run_status_label = ttk.Label(self.avg_run_config_frame, text="Ready.")
        self.avg_run_status_label.grid(row=8, column=0, columnspan=3, padx=5, pady=2, sticky="w")

        # --- Averaged Metrics Frame (Average Single Run Tab) ---
        self.avg_run_metrics_frame = ttk.LabelFrame(self.left_panel_avg_run, text="Averaged Performance Metrics")
        self.avg_run_metrics_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.avg_run_metrics_frame.grid_columnconfigure(1, weight=1)

        self.avg_run_metric_labels = {}
        # Metrics are the same as single prediction, but will be averaged
        for i, metric_name in enumerate(["Avg MSE", "Avg R2", "Avg MAPE", "Avg Accuracy (0-1)"]):
            ttk.Label(self.avg_run_metrics_frame, text=f"{metric_name}:").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.avg_run_metric_labels[metric_name] = ttk.Label(self.avg_run_metrics_frame, text="N/A")
            self.avg_run_metric_labels[metric_name].grid(row=i, column=1, padx=5, pady=2, sticky="ew")

        # Placeholder for plot in Average Single Run tab (currently empty)
        # We could add a plot showing distribution of metrics if desired later
        self.avg_run_plot_frame = ttk.LabelFrame(self.average_run_tab, text="Average Run Plot (Coming Soon)")
        self.avg_run_plot_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")


        # --- Tab 3: Experiment --- (Now tab 3)
        self.experiment_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.experiment_tab, text="Experiment")
        self.experiment_tab.grid_columnconfigure(0, weight=1)
        self.experiment_tab.grid_columnconfigure(1, weight=2)
        self.experiment_tab.grid_rowconfigure(0, weight=0)
        self.experiment_tab.grid_rowconfigure(1, weight=1)

        # --- Experiment Configuration (Experiment Tab) ---
        self.experiment_config_frame = ttk.LabelFrame(self.experiment_tab, text="Experiment Configuration")
        self.experiment_config_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.experiment_config_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.experiment_config_frame, text="Look Backs (comma-sep):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.experiment_lookbacks_var = tk.StringVar(value="30,60,90")
        ttk.Entry(self.experiment_config_frame, textvariable=self.experiment_lookbacks_var).grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        ttk.Label(self.experiment_config_frame, text="Number of Runs per LB:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.num_experiment_runs_var = tk.IntVar(value=5)
        ttk.Entry(self.experiment_config_frame, textvariable=self.num_experiment_runs_var).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.run_experiment_button = ttk.Button(self.experiment_config_frame, text="Run Experiment", command=self._run_experiment_thread)
        self.run_experiment_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        
        self.experiment_status_label = ttk.Label(self.experiment_config_frame, text="Ready.")
        self.experiment_status_label.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky="w")


        # --- Experiment Results Display (Experiment Tab) ---
        self.experiment_results_display_frame = ttk.LabelFrame(self.experiment_tab, text="Experiment Results (Average Metrics)")
        self.experiment_results_display_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.experiment_results_display_frame.grid_columnconfigure(0, weight=1)
        self.experiment_results_display_frame.grid_rowconfigure(0, weight=1)
        self.experiment_results_display_frame.grid_rowconfigure(1, weight=1)

        self.exp_results_tree = ttk.Treeview(self.experiment_results_display_frame, columns=("Look Back", "Avg MSE", "Avg R2", "Avg MAPE", "Avg Accuracy"), show="headings")
        self.exp_results_tree.heading("Look Back", text="Look Back")
        self.exp_results_tree.heading("Avg MSE", text="Avg MSE")
        self.exp_results_tree.heading("Avg R2", text="Avg R2")
        self.exp_results_tree.heading("Avg MAPE", text="Avg MAPE")
        self.exp_results_tree.heading("Avg Accuracy", text="Avg Accuracy")

        self.exp_results_tree.column("Look Back", width=100, anchor="center")
        self.exp_results_tree.column("Avg MSE", width=100, anchor="center")
        self.exp_results_tree.column("Avg R2", width=100, anchor="center")
        self.exp_results_tree.column("Avg MAPE", width=100, anchor="center")
        self.exp_results_tree.column("Avg Accuracy", width=100, anchor="center")

        self.exp_results_tree.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        tree_scrollbar = ttk.Scrollbar(self.experiment_results_display_frame, orient="vertical", command=self.exp_results_tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.exp_results_tree.configure(yscrollcommand=tree_scrollbar.set)

        self.fig_exp, self.ax_exp = plt.subplots(figsize=(10, 4))
        self.canvas_exp = FigureCanvasTkAgg(self.fig_exp, master=self.experiment_results_display_frame)
        self.canvas_exp_widget = self.canvas_exp.get_tk_widget()
        self.canvas_exp_widget.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self._clear_exp_plot()


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

    def _clear_exp_plot(self):
        self.ax_exp.clear()
        self.ax_exp.set_title('Experiment Summary: Average Metrics by Look Back')
        self.ax_exp.set_xlabel('Look Back Window')
        self.ax_exp.set_ylabel('Metric Value')
        self.ax_exp.grid(True)
        self.canvas_exp.draw()

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

    # --- Single Prediction Logic ---
    def _run_single_prediction_thread(self):
        # Disable all run buttons across tabs to prevent concurrent operations
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.status_label.config(text="Running single prediction... Please wait.")
        self.avg_run_status_label.config(text="")
        self.experiment_status_label.config(text="")
        self.master.update_idletasks()

        thread = threading.Thread(target=self._perform_single_prediction_task)
        thread.start()

    def _perform_single_prediction_task(self):
        try:
            metrics, plot_data = self._run_core_prediction_logic(
                ticker=self.ticker_var.get(),
                start_date_str=self.start_date_var.get(),
                end_date_str=self.end_date_var.get(),
                look_back=self.look_back_var.get(),
                train_test_split_ratio=self.train_ratio_var.get(),
                selected_model_name=self.model_var.get()
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
            self.ax_main.plot(plot_data['test_dates'], plot_data['actual_values'], label='Actual Values', color='black', linewidth=2)
            self.ax_main.plot(plot_data['test_dates'], plot_data['test_predict'], label=f'{self.model_var.get()} Predicted', color='purple', linestyle='--')
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

    # --- Core Prediction Logic (returns data, doesn't update GUI directly) ---
    def _run_core_prediction_logic(self, ticker, start_date_str, end_date_str, look_back, train_test_split_ratio, selected_model_name):
        # Input validation (duplicated to ensure this core function is robust)
        if not ticker or not start_date_str or not end_date_str:
            raise ValueError("All fields are required.")
        try:
            start_date_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_date_dt >= end_date_dt:
                raise ValueError("Start date must be before end date.")
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")
        if look_back <= 0:
            raise ValueError("Look back window must be a positive integer.")
        if not (0.1 <= train_test_split_ratio <= 0.99):
            raise ValueError("Train/Test ratio must be between 0.1 and 0.99.")

        data = get_stock_data_live(ticker, start_date_dt.strftime('%Y-%m-%d'), end_date_dt.strftime('%Y-%m-%d'))
        data = data.sort_index()
        features = ['Close']
        data_for_scaling = data[features].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_for_scaling)

        X, Y = create_sequences(scaled_data, look_back)
        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences with look-back {look_back}. Try reducing look-back or extending data range.")

        train_size = int(len(X) * train_test_split_ratio)
        if train_size < 1 or (len(X) - train_size) < 1:
            raise ValueError(f"Invalid train/test split for look-back {look_back}. Ensure at least one sample for training and one for testing. Adjust ratio or data range.")

        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        test_dates = data.index[train_size + look_back:]
        if len(test_dates) == 0:
            raise ValueError(f"No data available for the test (prediction) period for look-back {look_back}. Adjust date range or train/test ratio.")

        selected_model_builder = model_builders[selected_model_name]
        model = selected_model_builder((look_back, 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0,
                  callbacks=[early_stopping])

        test_predict_scaled = model.predict(X_test)

        dummy_predictions = np.zeros((len(test_predict_scaled), len(features)))
        dummy_predictions[:, 0] = test_predict_scaled.flatten()
        test_predict = scaler.inverse_transform(dummy_predictions)[:, 0]

        dummy_Y_test = np.zeros((len(Y_test), len(features)))
        dummy_Y_test[:, 0] = Y_test.flatten()
        actual_values = scaler.inverse_transform(dummy_Y_test)[:, 0]

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
            'test_predict': test_predict
        }

        return metrics_results, plot_data

    # --- Average Single Run Logic ---
    def _run_average_prediction_thread(self):
        # Disable all run buttons across tabs
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.avg_run_status_label.config(text="Running average prediction... Please wait.")
        self.status_label.config(text="")
        self.experiment_status_label.config(text="")
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

            for run_idx in range(num_runs):
                self.avg_run_status_label.config(text=f"Running average prediction: Run {run_idx+1}/{num_runs}...")
                self.master.update_idletasks()
                
                try:
                    metrics, _ = self._run_core_prediction_logic(
                        ticker=self.avg_run_ticker_var.get(),
                        start_date_str=self.avg_run_start_date_var.get(),
                        end_date_str=self.avg_run_end_date_var.get(),
                        look_back=self.avg_run_look_back_var.get(),
                        train_test_split_ratio=self.avg_run_train_ratio_var.get(),
                        selected_model_name=self.avg_run_model_var.get()
                    )
                    all_mses.append(metrics['MSE'])
                    all_r2s.append(metrics['R2'])
                    all_mapes.append(metrics['MAPE'])
                    all_accuracies.append(metrics['Accuracy (0-1)'])
                except Exception as e:
                    print(f"Error during average run {run_idx+1}: {e}")
                    messagebox.showwarning("Average Run Error", f"Failed during run {run_idx+1}: {e}. Skipping this run.")
                    # Continue to next run if one fails, but don't include failed metrics

            if all_mses: # Ensure we have at least one successful run
                avg_mse = np.mean(all_mses)
                avg_r2 = np.mean(all_r2s)
                avg_mape = np.mean(all_mapes)
                avg_accuracy = np.mean(all_accuracies)

                self.avg_run_metric_labels["Avg MSE"].config(text=f"{avg_mse:.4f}")
                self.avg_run_metric_labels["Avg R2"].config(text=f"{avg_r2:.4f}")
                self.avg_run_metric_labels["Avg MAPE"].config(text=f"{avg_mape:.2f}%")
                self.avg_run_metric_labels["Avg Accuracy (0-1)"].config(text=f"{avg_accuracy:.4f}")
                self.avg_run_status_label.config(text=f"Average prediction complete over {len(all_mses)} successful runs!")
            else:
                self.avg_run_status_label.config(text="No successful runs to average. Check configurations.")
                for label in self.avg_run_metric_labels.values():
                    label.config(text="N/A")


        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.avg_run_status_label.config(text="Error occurred during average prediction.")
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)

    # --- Experiment Logic ---
    def _run_experiment_thread(self):
        # Disable all run buttons across tabs
        self.run_button.config(state=tk.DISABLED)
        self.run_avg_button.config(state=tk.DISABLED)
        self.run_experiment_button.config(state=tk.DISABLED)
        self.experiment_status_label.config(text="Running experiment... Please wait. This may take a while.")
        self.status_label.config(text="")
        self.avg_run_status_label.config(text="")
        self.master.update_idletasks()

        thread = threading.Thread(target=self._perform_experiment_task)
        thread.start()

    def _perform_experiment_task(self):
        try:
            look_backs_str = self.experiment_lookbacks_var.get()
            num_runs = self.num_experiment_runs_var.get()

            try:
                look_backs = [int(lb.strip()) for lb in look_backs_str.split(',') if lb.strip()]
                if not look_backs:
                    raise ValueError("Please enter at least one look back value for the experiment.")
                for lb in look_backs:
                    if lb <= 0:
                        raise ValueError("Look back values for experiment must be positive integers.")
            except ValueError:
                raise ValueError("Invalid look back values for experiment. Enter comma-separated integers (e.g., '30,60,90').")

            if num_runs <= 0:
                raise ValueError("Number of runs per look back must be a positive integer.")

            all_experiment_averages = []

            for item in self.exp_results_tree.get_children():
                self.exp_results_tree.delete(item)
            self._clear_exp_plot()

            # Ensure these are read from the single prediction tab's values,
            # as the experiment doesn't have its own specific Ticker/Date inputs
            current_ticker = self.ticker_var.get()
            current_start_date = self.start_date_var.get()
            current_end_date = self.end_date_var.get()
            current_train_ratio = self.train_ratio_var.get()
            current_model_name = self.model_var.get()

            for i, current_look_back in enumerate(look_backs):
                individual_run_metrics = []
                for run_idx in range(num_runs):
                    self.experiment_status_label.config(text=f"Experiment: LB {current_look_back}, Run {run_idx+1}/{num_runs}. Model: {current_model_name}")
                    self.master.update_idletasks()
                    
                    try:
                        metrics, _ = self._run_core_prediction_logic(
                            ticker=current_ticker,
                            start_date_str=current_start_date,
                            end_date_str=current_end_date,
                            look_back=current_look_back,
                            train_test_split_ratio=current_train_ratio,
                            selected_model_name=current_model_name
                        )
                        individual_run_metrics.append({
                            'MSE': metrics['MSE'],
                            'R2': metrics['R2'],
                            'MAPE': metrics['MAPE'],
                            'Accuracy (0-1)': metrics['Accuracy (0-1)']
                        })
                    except Exception as e:
                        print(f"Error during experiment run (LB: {current_look_back}, Run: {run_idx+1}): {e}")
                        messagebox.showwarning("Experiment Run Error", f"Failed for Look Back {current_look_back}, Run {run_idx+1}: {e}. Continuing experiment.")

                if individual_run_metrics:
                    avg_mse = np.mean([r['MSE'] for r in individual_run_metrics])
                    avg_r2 = np.mean([r['R2'] for r in individual_run_metrics])
                    avg_mape = np.mean([r['MAPE'] for r in individual_run_metrics])
                    avg_accuracy = np.mean([r['Accuracy (0-1)'] for r in individual_run_metrics])

                    all_experiment_averages.append({
                        'Look Back': current_look_back,
                        'Avg MSE': avg_mse,
                        'Avg R2': avg_r2,
                        'Avg MAPE': avg_mape,
                        'Avg Accuracy': avg_accuracy
                    })
                    
                    self.exp_results_tree.insert("", "end", values=(
                        current_look_back,
                        f"{avg_mse:.4f}",
                        f"{avg_r2:.4f}",
                        f"{avg_mape:.2f}%",
                        f"{avg_accuracy:.4f}"
                    ))
                else:
                    self.exp_results_tree.insert("", "end", values=(current_look_back, "N/A", "N/A", "N/A", "N/A"))

            all_experiment_averages.sort(key=lambda x: x['Look Back'])

            if all_experiment_averages:
                look_backs_plotted = [r['Look Back'] for r in all_experiment_averages]
                avg_mses = [r['Avg MSE'] for r in all_experiment_averages]
                avg_r2s = [r['Avg R2'] for r in all_experiment_averages]

                self.ax_exp.clear()
                
                ax_r2 = self.ax_exp.twinx()

                bar_width = 0.35
                r1 = np.arange(len(look_backs_plotted))
                r2 = [x + bar_width for x in r1]

                self.ax_exp.bar(r1, avg_mses, color='skyblue', width=bar_width, edgecolor='grey', label='Avg MSE')
                ax_r2.bar(r2, avg_r2s, color='lightcoral', width=bar_width, edgecolor='grey', label='Avg R2')
                
                for i in range(len(r1)):
                    self.ax_exp.text(r1[i], avg_mses[i], f'{avg_mses[i]:.2f}', ha='center', va='bottom', fontsize=8)
                    ax_r2.text(r2[i], avg_r2s[i], f'{avg_r2s[i]:.2f}', ha='center', va='bottom', fontsize=8)

                self.ax_exp.set_title(f'Experiment Summary: Average Metrics by Look Back ({current_model_name})')
                self.ax_exp.set_xlabel('Look Back Window')
                self.ax_exp.set_ylabel('Average MSE', color='skyblue')
                ax_r2.set_ylabel('Average R2', color='lightcoral')
                
                self.ax_exp.set_xticks([r + bar_width/2 for r in range(len(look_backs_plotted))])
                self.ax_exp.set_xticklabels(look_backs_plotted)
                
                lines, labels = self.ax_exp.get_legend_handles_labels()
                lines2, labels2 = ax_r2.get_legend_handles_labels()
                ax_r2.legend(lines + lines2, labels + labels2, loc='best')

                self.ax_exp.grid(True)
                self.fig_exp.tight_layout()
                self.canvas_exp.draw()

            self.experiment_status_label.config(text="Experiment complete!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.experiment_status_label.config(text="Error occurred during experiment.")
            self._clear_exp_plot()
        finally:
            self.run_button.config(state=tk.NORMAL)
            self.run_avg_button.config(state=tk.NORMAL)
            self.run_experiment_button.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()