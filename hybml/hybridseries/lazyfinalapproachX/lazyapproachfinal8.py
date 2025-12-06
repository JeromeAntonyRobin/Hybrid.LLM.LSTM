import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import yfinance as yf
import pandas as pd
import numpy as np
import threading
from datetime import datetime, timedelta
import re
import traceback
import os
import webbrowser
import json
import time  # For simulating delays

# Plotly imports
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots  # Import for subplots

# Sentiment analysis libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Machine learning libraries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Ollama library - ensure Ollama server is running and 'deepseek-r1' model is pulled
import ollama


class ForwardLookingStockPredictor:
    """
    A Tkinter application for predicting stock prices using historical data,
    and influencing that prediction with recent news sentiment (VADER, TextBlob, or DeepSeek-R1).
    The application features a tabbed UI for prediction results, detailed news sentiment,
    technical indicator visualizations, and DeepSeek-R1 interaction logs.
    It predicts multiple future trading days and provides confidence intervals.
    Includes enhanced DeepSeek-R1 integration for news relevance filtering, dynamic prediction influence,
    macroeconomic outlook, automated company fundamental outlook, specific risk factor identification,
    and news summarization. Graphs are generated using Plotly and opened in a web browser.
    DeepSeek-R1 responses are now requested in JSON format for dynamic adjustments, with robust fallback parsing.
    LSTM now incorporates more features (Open, High, Low, Volume, Technical Indicators).
    MAX_ADJUSTMENT_PERCENT values have been adjusted to better reflect market reactions.
    """

    # Define max adjustment percentages for dynamic scaling. These are tunable.
    # Increased values slightly to give more weight to sentiment and risk based on user feedback.
    MAX_MACRO_ADJUSTMENT_PERCENT = 0.002  # Max 0.2% adjustment based on macro strength
    MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT = 0.003  # Max 0.3% adjustment based on fundamental strength
    MAX_SENTIMENT_ADJUSTMENT_PERCENT = 0.008  # Increased from 0.005 to 0.008 (Max 0.8% adjustment based on sentiment score)
    MAX_RISK_ADJUSTMENT_PERCENT = 0.015  # Increased from 0.010 to 0.015 (Max 1.5% downward adjustment based on total risk severity)

    def __init__(self, master):
        self.master = master
        master.title("Forward-Looking Stock Predictor with Enhanced Intelligence")
        master.geometry("1400x950")  # Larger window for tabs and logs
        master.resizable(True, True)

        self._setup_ui()

        # Initialize analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.plotly_html_file = "stock_prediction_plot.html"
        self.plotly_indicators_html_file = "stock_indicators_plot.html"  # New file for indicators plot

        # Caches for display
        self.cached_news_sentiment_details = []
        self.cached_overall_sentiment = 0.0
        self.cached_macro_outlook = "N/A"
        self.cached_company_fundamental_outlook = "N/A"
        self.cached_risk_factors = []
        self.cached_fundamental_metrics_used = {}
        self.cached_llm_interactions_log = ""
        self.cached_market_insight_summary = "No market insight available yet."  # New cache for market insight

        # Initial Ollama status check in a separate thread
        self._start_ollama_status_check()

    def _setup_ui(self):
        # Main Notebook (Tabbed Interface)
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 1: Prediction & Plot ---
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="Prediction & Plot")
        self.prediction_tab.grid_columnconfigure(0, weight=1)  # Left panel (inputs, logs)
        self.prediction_tab.grid_columnconfigure(1, weight=1)  # Right panel (plot button area)
        self.prediction_tab.grid_rowconfigure(0, weight=0)  # Input frame fixed size
        self.prediction_tab.grid_rowconfigure(1, weight=1)  # Logs and Plotly button area expand vertically

        # --- Input Parameters Frame ---
        self.input_frame = tk.LabelFrame(self.prediction_tab, text="Prediction Parameters", padx=10, pady=10)
        self.input_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Configure input_frame grid for 2 columns of inputs
        self.input_frame.grid_columnconfigure(1, weight=1)  # Makes the entry fields expand
        self.input_frame.grid_columnconfigure(3, weight=1)  # Makes the entry fields expand

        row_idx = 0
        tk.Label(self.input_frame, text="Company Ticker (e.g., AAPL):").grid(row=row_idx, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.ticker_entry = tk.Entry(self.input_frame)
        self.ticker_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        self.ticker_entry.insert(0, "AAPL")  # Default value

        tk.Label(self.input_frame, text="Historical Data Start (YYYY-MM-DD):").grid(row=row_idx, column=2, padx=5,
                                                                                    pady=5, sticky="w")
        self.start_date_entry = tk.Entry(self.input_frame)
        self.start_date_entry.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d'))
        row_idx += 1

        tk.Label(self.input_frame, text="Historical Data End (YYYY-MM-DD):").grid(row=row_idx, column=0, padx=5, pady=5,
                                                                                  sticky="w")
        self.end_date_entry = tk.Entry(self.input_frame)
        self.end_date_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))

        tk.Label(self.input_frame, text="LSTM Lookback Window (days):").grid(row=row_idx, column=2, padx=5, pady=5,
                                                                             sticky="w")
        self.lookback_entry = tk.Entry(self.input_frame)
        self.lookback_entry.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")
        self.lookback_entry.insert(0, "60")
        row_idx += 1

        # New: Prediction Days input
        tk.Label(self.input_frame, text="Prediction Days (1-10):").grid(row=row_idx, column=0, padx=5, pady=5,
                                                                        sticky="w")
        self.prediction_days_entry = tk.Entry(self.input_frame)
        self.prediction_days_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        self.prediction_days_entry.insert(0, "2")  # Default to 2 days

        tk.Label(self.input_frame, text="Sentiment Analyzer:").grid(row=row_idx, column=2, padx=5, pady=5, sticky="w")
        self.analyzer_choice = ttk.Combobox(self.input_frame,
                                            values=["DeepSeek-R1", "VADER", "TextBlob"],
                                            state="readonly")
        self.analyzer_choice.set("DeepSeek-R1")
        self.analyzer_choice.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")
        row_idx += 1

        self.predict_button = tk.Button(self.input_frame, text="Predict Prices & Generate Plots",
                                        command=self._start_prediction_thread)
        self.predict_button.grid(row=row_idx, column=0, columnspan=4, pady=10, sticky="ew")
        row_idx += 1

        self.ollama_status_label = tk.Label(self.input_frame, text="Ollama Status: Checking...", fg="gray")
        self.ollama_status_label.grid(row=row_idx, column=0, columnspan=4, pady=5, sticky="ew")

        # --- Left Panel: Prediction Results and Logs ---
        self.left_panel_frame = tk.Frame(self.prediction_tab)
        self.left_panel_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.left_panel_frame.grid_rowconfigure(0, weight=0)  # Prediction label fixed size
        self.left_panel_frame.grid_rowconfigure(1, weight=1)  # Log display expands
        self.left_panel_frame.grid_columnconfigure(0, weight=1)

        self.prediction_label = tk.Label(self.left_panel_frame, text="Prediction: Awaiting inputs...", wraplength=480,
                                         justify=tk.LEFT, font=("Arial", 11, "bold"))
        self.prediction_label.grid(row=0, column=0, padx=5, pady=5, sticky="nw")

        self.log_frame = tk.LabelFrame(self.left_panel_frame, text="Processing Logs", padx=5, pady=5)
        self.log_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)

        self.log_text_display = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, width=60, height=15,
                                                          state='disabled')
        self.log_text_display.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # --- Right Panel: Plotly Buttons Frame ---
        self.plot_controls_frame = tk.LabelFrame(self.prediction_tab, text="View Plots", padx=5, pady=5)
        self.plot_controls_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.plot_controls_frame.grid_rowconfigure(0, weight=1)  # Spacer for vertical alignment
        self.plot_controls_frame.grid_rowconfigure(1, weight=0)  # Button row
        self.plot_controls_frame.grid_rowconfigure(2, weight=0)  # Second button row
        self.plot_controls_frame.grid_columnconfigure(0, weight=1)

        self.open_plotly_button = tk.Button(self.plot_controls_frame, text="Open Interactive Prediction Plot",
                                            command=self._open_plotly_html, state=tk.DISABLED)
        self.open_plotly_button.grid(row=1, column=0, padx=10, pady=5, sticky="s")  # Stick to bottom

        self.open_indicators_button = tk.Button(self.plot_controls_frame, text="Open Technical Indicators Plot",
                                                command=self._open_indicators_html, state=tk.DISABLED)
        self.open_indicators_button.grid(row=2, column=0, padx=10, pady=5, sticky="s")  # Below prediction button

        # --- Tab 2: News & Sentiment Details ---
        self.news_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.news_tab, text="Recent News & Market Insights")
        self.news_tab.grid_columnconfigure(0, weight=1)
        self.news_tab.grid_rowconfigure(0, weight=0)  # Market insight summary
        self.news_tab.grid_rowconfigure(1, weight=1)  # News details

        self.market_insight_label = tk.Label(self.news_tab, text="Overall Market Insight: No insight available yet.",
                                             wraplength=700, justify=tk.LEFT, font=("Arial", 12, "bold"), fg="darkblue")
        self.market_insight_label.grid(row=0, column=0, padx=10, pady=(10, 10), sticky="nw")

        self.news_details_text = scrolledtext.ScrolledText(self.news_tab, wrap=tk.WORD, width=100, height=30,
                                                           state='disabled')
        self.news_details_text.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # --- Tab 3: Technical Indicators (new tab) ---
        self.indicators_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.indicators_tab, text="Technical Indicators")
        self.indicators_tab.grid_columnconfigure(0, weight=1)
        self.indicators_tab.grid_rowconfigure(0, weight=1)

        tk.Label(self.indicators_tab, text="Technical Indicator plots will open in your web browser.",
                 font=("Arial", 12, "italic")).grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- Tab 4: DeepSeek-R1 Interactions ---
        self.llm_interactions_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.llm_interactions_tab, text="DeepSeek-R1 Logs")
        self.llm_interactions_tab.grid_columnconfigure(0, weight=1)
        self.llm_interactions_tab.grid_rowconfigure(0, weight=1)

        self.llm_log_text = scrolledtext.ScrolledText(self.llm_interactions_tab, wrap=tk.WORD, state='disabled')
        self.llm_log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- Status Bar (at bottom of main window) ---
        self.status_label = tk.Label(self.master, text="Ready.", relief=tk.SUNKEN, bd=1, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _show_custom_messagebox(self, title, message, message_type="info"):
        """Custom message box using Tkinter Toplevel."""
        top_level = tk.Toplevel(self.master)
        top_level.title(title)
        # top_level.overrideredirect(True) # Can uncomment if you want to remove title bar, but might look less native

        # Center the Toplevel window
        self.master.update_idletasks()  # Ensure main window geometry is updated
        x = self.master.winfo_x() + (self.master.winfo_width() // 2) - (400 // 2)  # 400 is dialog width
        y = self.master.winfo_y() + (self.master.winfo_height() // 2) - (200 // 2)  # 200 is dialog height
        top_level.geometry(f"400x200+{x}+{y}")  # Set size and position

        top_level.transient(self.master)  # Make it modal relative to main window
        top_level.grab_set()  # Grab all events until it's destroyed

        label_color = "black"
        if message_type == "error":
            label_color = "red"
        elif message_type == "warning":
            label_color = "orange"
        elif message_type == "success":
            label_color = "green"
        elif message_type == "info":
            label_color = "blue"

        # Frame for content and padding
        content_frame = tk.Frame(top_level, padx=10, pady=10)
        content_frame.pack(expand=True, fill="both")
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        tk.Label(content_frame, text=message, wraplength=350, justify="center",
                 font=("Arial", 12, "bold"), fg=label_color).grid(row=0, column=0, pady=(20, 10), padx=10,
                                                                  sticky="nsew")

        tk.Button(content_frame, text="OK", command=top_level.destroy).grid(row=1, column=0, pady=(0, 10))

        self.master.wait_window(top_level)  # Block until the Toplevel window is closed

    def _update_status(self, message, color="black"):
        """Helper to update the status bar label."""
        self.status_label.config(text=message, fg=color)
        self.master.update_idletasks()

    def _update_log_display(self, text_to_add, clear_previous=True):
        """Helper to safely update the scrolledtext widget for general logs."""
        self.log_text_display.config(state='normal')
        if clear_previous:
            self.log_text_display.delete(1.0, tk.END)
        self.log_text_display.insert(tk.END, text_to_add + "\n")
        self.log_text_display.see(tk.END)
        self.log_text_display.config(state='disabled')

    def _update_llm_log_display(self, text_to_add, append=True):
        """Helper to safely update the scrolledtext widget for LLM interaction logs."""
        self.llm_log_text.config(state='normal')
        if not append:
            self.llm_log_text.delete(1.0, tk.END)
            self.cached_llm_interactions_log = ""  # Clear cached log too

        self.llm_log_text.insert(tk.END, text_to_add + "\n")
        self.cached_llm_interactions_log += text_to_add + "\n"  # Append to cache
        self.llm_log_text.see(tk.END)
        self.llm_log_text.config(state='disabled')

    def _update_news_details_display(self):
        """Helper to update the scrolledtext widget for news details."""
        self.news_details_text.config(state='normal')
        self.news_details_text.delete(1.0, tk.END)

        self.market_insight_label.config(text=f"Overall Market Insight: {self.cached_market_insight_summary}")

        if not self.cached_news_sentiment_details and not self.cached_risk_factors:
            self.news_details_text.insert(tk.END,
                                          "No recent relevant news found or analyzed for the last prediction run.\n")
        else:
            self.news_details_text.insert(tk.END,
                                          f"Overall Average Sentiment Score (from relevant news): {self.cached_overall_sentiment:.4f}\n")
            self.news_details_text.insert(tk.END,
                                          f"Identified Specific Risks (from relevant news): {', '.join(self.cached_risk_factors) if self.cached_risk_factors else 'None'}\n\n")

            if self.cached_news_sentiment_details:
                for item in self.cached_news_sentiment_details:
                    # Use .get() with default empty string to prevent KeyError
                    title = item.get('title', 'No Title Provided')
                    summary = item.get('summary', 'No Summary Provided')
                    deepseek_summary = item.get('deepseek_summary', '')  # Get DeepSeek-R1 summary if present

                    self.news_details_text.insert(tk.END, f"Title: {title}\n")
                    self.news_details_text.insert(tk.END, f"Summary: {summary}\n")
                    if deepseek_summary:  # Only show if DeepSeek-R1 summary exists
                        self.news_details_text.insert(tk.END, f"DeepSeek-R1 News Insight: {deepseek_summary}\n")
                    self.news_details_text.insert(tk.END,
                                                  f"Individual Sentiment ({item['analyzer']}): {item['sentiment']:.4f}\n")
                    item_risks_display = f"Identified Risks in this news: {', '.join(item['risks'])} (Severity: {item['risk_severity']:.1f})" if \
                        item['risks'] else 'Identified Risks in this news: None'
                    self.news_details_text.insert(tk.END, f"{item_risks_display}\n")
                    self.news_details_text.insert(tk.END, "---\n\n")
            else:
                self.news_details_text.insert(tk.END,
                                              "No relevant news items found to display detailed sentiment or risks.\n")

        self.news_details_text.see(tk.END)
        self.news_details_text.config(state='disabled')

    def _open_plotly_html(self):
        """Opens the generated Plotly HTML file in the default web browser."""
        if os.path.exists(self.plotly_html_file):
            webbrowser.open_new_tab(f"file:///{os.path.abspath(self.plotly_html_file)}")
            self._update_status(f"Opened {self.plotly_html_file} in your web browser.", "blue")
        else:
            self._show_custom_messagebox("File Not Found",
                                         "Prediction plot HTML file not found. Please run a prediction first.", "error")
            self._update_status("Plotly HTML file not found.", "red")

    def _open_indicators_html(self):
        """Opens the generated Plotly HTML file for indicators in the default web browser."""
        if os.path.exists(self.plotly_indicators_html_file):
            webbrowser.open_new_tab(f"file:///{os.path.abspath(self.plotly_indicators_html_file)}")
            self._update_status(f"Opened {self.plotly_indicators_html_file} in your web browser.", "blue")
        else:
            self._show_custom_messagebox("File Not Found",
                                         "Indicators plot HTML file not found. Please run a prediction first.", "error")
            self._update_status("Indicators Plotly HTML file not found.", "red")

    def _start_ollama_status_check(self):
        """Starts a thread to check Ollama status."""
        thread = threading.Thread(target=self._check_ollama_status_task)
        thread.daemon = True
        thread.start()

    def _check_ollama_status_task(self):
        """Task to check Ollama server status and update GUI."""
        try:
            response = ollama.chat(model='deepseek-r1', messages=[{'role': 'user', 'content': 'hello'}], stream=True,
                                   options={'num_predict': 1})
            for chunk in response:
                pass
            self.master.after(0, lambda: self.ollama_status_label.config(text="Ollama Status: Running (deepseek-r1)",
                                                                         fg="green"))
        except Exception as e:
            self.master.after(0, lambda: self.ollama_status_label.config(
                text=f"Ollama Status: Not Running or Error ({e}). Ensure 'deepseek-r1' model is pulled.", fg="red"))

    def _start_prediction_thread(self):
        """Starts a new thread for the entire prediction process."""
        ticker = self.ticker_entry.get().strip().upper()
        start_date_str = self.start_date_entry.get()
        end_date_str = self.end_date_entry.get()
        lookback_window_str = self.lookback_entry.get()
        prediction_days_str = self.prediction_days_entry.get()  # New: Get prediction days
        selected_analyzer = self.analyzer_choice.get()

        if not all([ticker, start_date_str, end_date_str, lookback_window_str, prediction_days_str,
                    selected_analyzer]):
            self._show_custom_messagebox("Input Error", "Please fill in all required fields.", "warning")
            return

        try:
            lookback_window = int(lookback_window_str)
            prediction_days = int(prediction_days_str)  # New: Convert to int
            if not (1 <= prediction_days <= 10):  # Validate prediction days
                raise ValueError("Number of prediction days must be between 1 and 10.")
            datetime.strptime(start_date_str, '%Y-%m-%d')
            datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError as ve:
            self._show_custom_messagebox("Input Error",
                                         f"Invalid input: {ve}\nLookback window and prediction days must be integers. Dates must beYYYY-MM-DD.",
                                         "warning")
            return

        self.predict_button.config(state=tk.DISABLED)
        self.open_plotly_button.config(state=tk.DISABLED)
        self.open_indicators_button.config(state=tk.DISABLED)  # Disable indicators button too
        self._update_status("Starting prediction process...", "blue")
        self._update_log_display("", clear_previous=True)
        self._update_llm_log_display("", append=False)  # Clear LLM log tab
        self.prediction_label.config(text="Prediction: Processing...")
        self.market_insight_label.config(text="Overall Market Insight: Processing...")  # Update market insight label

        # Clear caches
        self.cached_news_sentiment_details = []
        self.cached_overall_sentiment = 0.0
        self.cached_macro_outlook = "N/A"
        self.cached_company_fundamental_outlook = "N/A"
        self.cached_risk_factors = []
        self.cached_fundamental_metrics_used = {}
        self.cached_market_insight_summary = "No market insight available yet."  # Clear market insight cache
        self.master.after(0, self._update_news_details_display)  # Update news tab to clear it

        thread = threading.Thread(target=self._perform_prediction_task,
                                  args=(ticker, start_date_str, end_date_str,
                                        lookback_window, prediction_days,  # Pass prediction_days
                                        selected_analyzer))
        thread.daemon = True
        thread.start()

    def _perform_prediction_task(self, ticker, start_date_str, end_date_str, lookback_window,
                                 prediction_days, selected_analyzer):  # Added prediction_days
        """
        Main task to fetch data, analyze sentiment, train LSTM, and predict.
        Runs in a separate thread.
        """
        try:
            start_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_date_dt >= end_date_dt:
                raise ValueError("Start date must be before end date.")

            # --- Step 1: Retrieve Historical Stock Data ---
            self._update_status(
                f"Downloading historical stock data for {ticker} from {start_date_str} to {end_date_str}...", "blue")
            ticker_yf_obj = yf.Ticker(ticker)
            stock_data = ticker_yf_obj.history(start=start_date_str, end=end_date_dt, auto_adjust=False)
            if stock_data.empty:
                raise ValueError(
                    f"No stock data found for {ticker} in the specified date range. Please check ticker or dates.")

            if 'Adj Close' in stock_data.columns:
                stock_data['Close'] = stock_data['Adj Close']

            ohlcv_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            ohlcv_data.index = pd.to_datetime(ohlcv_data.index).normalize()
            ohlcv_data.index.name = 'date'
            ohlcv_data.sort_index(inplace=True)

            self._update_log_display(f"Raw OHLCV data head:\n{ohlcv_data.head().to_string()}\n")

            # --- Calculate Technical Indicators ---
            self._update_status("Calculating technical indicators...", "blue")

            ohlcv_data['SMA_10'] = ohlcv_data['Close'].rolling(window=10).mean()
            ohlcv_data['SMA_20'] = ohlcv_data['Close'].rolling(window=20).mean()

            delta = ohlcv_data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(span=14, adjust=False).mean()
            avg_loss = loss.ewm(span=14, adjust=False).mean()

            rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
            ohlcv_data['RSI'] = 100 - (100 / (1 + rs))

            exp1 = ohlcv_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = ohlcv_data['Close'].ewm(span=26, adjust=False).mean()
            ohlcv_data['MACD'] = exp1 - exp2
            ohlcv_data['Signal_Line'] = ohlcv_data['MACD'].ewm(span=9, adjust=False).mean()
            ohlcv_data['MACD_Hist'] = ohlcv_data['MACD'] - ohlcv_data['Signal_Line']

            initial_rows_count = len(ohlcv_data)
            ohlcv_data.dropna(inplace=True)
            if len(ohlcv_data) < initial_rows_count:
                self._update_log_display(
                    f"Dropped {initial_rows_count - len(ohlcv_data)} rows due to NaN values from technical indicator calculations.\n",
                    clear_previous=False)

            if ohlcv_data.empty:
                raise ValueError(
                    "Not enough data to calculate technical indicators and train LSTM after dropping NaN values. Try a longer historical period.")

            self._update_log_display(f"Data with indicators head:\n{ohlcv_data.head().to_string()}\n",
                                     clear_previous=False)
            self._update_log_display(f"Data with indicators tail:\n{ohlcv_data.tail().to_string()}\n",
                                     clear_previous=False)

            company_long_name = ticker_yf_obj.info.get('longName', ticker)

            # --- DeepSeek-R1 for Macroeconomic Outlook ---
            self._update_status("DeepSeek-R1: Assessing overall macroeconomic outlook...", "blue")
            macro_prompt = (
                "Considering general global economic conditions and recent news trends (e.g., inflation, interest rates, GDP, geopolitical events), "
                "classify the current short-term (next 1-2 weeks) macroeconomic outlook for the stock market. "
                "Respond with a JSON object like this: `{\"outlook\": \"Bullish\"}`. "
                "The value for \"outlook\" should be one word: 'Bullish', 'Neutral', or 'Bearish'. No other text or filler."
            )
            self._update_llm_log_display(f"--- Macroeconomic Outlook ---\nPrompt:\n{macro_prompt}\n")
            macro_outlook_classification = "Neutral"  # Default
            macro_strength = 0.0  # Default strength for neutral or parsing failure

            try:
                macro_response = ollama.chat(
                    model='deepseek-r1', messages=[{'role': 'user', 'content': macro_prompt}], stream=False,
                    options={'num_predict': 20}, format='json'
                )
                macro_raw_answer = macro_response['message']['content'].strip()

                parsed_outlook = "Neutral"  # Default
                try:
                    json_data = json.loads(macro_raw_answer)
                    if 'outlook' in json_data:
                        parsed_outlook = json_data['outlook'].strip().capitalize()
                except json.JSONDecodeError:
                    self._update_llm_log_display(
                        f"Macro Outlook: JSON parse failed. Raw: {macro_raw_answer}. Falling back to regex.\n")
                    if re.search(r"bullish", macro_raw_answer, re.IGNORECASE):
                        parsed_outlook = "Bullish"
                    elif re.search(r"bearish", macro_raw_answer, re.IGNORECASE):
                        parsed_outlook = "Bearish"
                    elif re.search(r"neutral", macro_raw_answer, re.IGNORECASE):
                        parsed_outlook = "Neutral"

                macro_outlook_classification = parsed_outlook

                # If a specific outlook (Bullish/Bearish) is given, ask for strength
                if macro_outlook_classification in ["Bullish", "Bearish"]:
                    strength_prompt = (
                        f"On a scale of 0.0 to 1.0 (0.0=no strength, 1.0=very strong), "
                        f"how strong is this '{macro_outlook_classification}' macroeconomic outlook? "
                        "Respond with a JSON object like this: `{\"strength\": 0.7}`. No other text or filler."
                    )
                    self._update_llm_log_display(f"--- Macroeconomic Strength ---\nPrompt:\n{strength_prompt}\n")
                    strength_response = ollama.chat(
                        model='deepseek-r1', messages=[{'role': 'user', 'content': strength_prompt}], stream=False,
                        options={'num_predict': 20}, format='json'
                    )
                    strength_raw_answer = strength_response['message']['content'].strip()
                    self._update_llm_log_display(f"Raw Response: {strength_raw_answer}\n")
                    try:
                        json_data = json.loads(strength_raw_answer)
                        if 'strength' in json_data:
                            macro_strength = float(json_data['strength'])
                            macro_strength = max(0.0, min(1.0, macro_strength))  # Clamp between 0 and 1
                    except (json.JSONDecodeError, ValueError):
                        self._update_llm_log_display(
                            f"Macro Strength: JSON parse failed. Raw: {strength_raw_answer}. Falling back to regex.\n")
                        score_match = re.search(r"([-+]?\d*\.?\d+)", strength_raw_answer)
                        if score_match:
                            macro_strength = max(0.0, min(1.0, float(score_match.group(0))))

                self._update_llm_log_display(
                    f"Parsed Value: '{macro_outlook_classification}', Strength: {macro_strength:.2f}\n")

            except Exception as e:
                self._update_log_display(
                    f"Warning: DeepSeek-R1 macro outlook/strength failed: {e}. Defaulting to Neutral with 0 strength.",
                    clear_previous=False)
                self._update_llm_log_display(f"Error: {e}. Defaulting to Neutral with 0 strength.\n")
                macro_outlook_classification = "Neutral"
                macro_strength = 0.0

            self.cached_macro_outlook = macro_outlook_classification  # Cache for display
            macro_bias_magnitude = macro_strength * self.MAX_MACRO_ADJUSTMENT_PERCENT
            self._update_log_display(
                f"DeepSeek-R1 Macroeconomic Outlook: {macro_outlook_classification} (Bias Magnitude: {macro_bias_magnitude * 100:.2f}%)\n",
                clear_previous=False)

            # --- Automated DeepSeek-R1 for Company Fundamental Outlook ---
            self._update_status(f"DeepSeek-R1: Assessing {ticker} company fundamental outlook...", "blue")

            company_info = ticker_yf_obj.info
            fundamental_metrics = {}

            if 'revenuePerShare' in company_info and company_info['revenuePerShare'] is not None:
                fundamental_metrics['Revenue Per Share'] = company_info['revenuePerShare']
            if 'profitMargins' in company_info and company_info['profitMargins'] is not None:
                fundamental_metrics['Profit Margins'] = f"{company_info['profitMargins'] * 100:.2f}%"
            elif 'grossMargins' in company_info and company_info['grossMargins'] is not None:
                fundamental_metrics['Gross Margins'] = f"{company_info['grossMargins'] * 100:.2f}%"
            elif 'operatingMargins' in company_info and company_info['operatingMargins'] is not None:
                fundamental_metrics['Operating Margins'] = f"{company_info['operatingMargins'] * 100:.2f}%"

            if 'currentRatio' in company_info and company_info['currentRatio'] is not None:
                fundamental_metrics['Current Ratio'] = company_info['currentRatio']
            if 'debtToEquity' in company_info and company_info['debtToEquity'] is not None:
                fundamental_metrics['Debt to Equity'] = company_info['debtToEquity']
            if 'returnOnEquity' in company_info and company_info['returnOnEquity'] is not None:
                fundamental_metrics['Return on Equity'] = f"{company_info['returnOnEquity'] * 100:.2f}%"
            if 'forwardPE' in company_info and company_info['forwardPE'] is not None:
                fundamental_metrics['Forward P/E'] = company_info['forwardPE']
            if 'pegRatio' in company_info and company_info['pegRatio'] is not None:
                fundamental_metrics['PEG Ratio'] = company_info['pegRatio']
            if 'beta' in company_info and company_info['beta'] is not None:
                fundamental_metrics['Beta'] = company_info['beta']

            fundamental_data_str = "No specific fundamental data found via yfinance.info."
            if fundamental_metrics:
                fundamental_data_str = ", ".join([f"{k}: {v}" for k, v in fundamental_metrics.items()])

            self.cached_fundamental_metrics_used = fundamental_metrics  # Cache for display

            fundamental_prompt = (
                f"Analyze the fundamental health and outlook for {company_long_name} ({ticker}) based on these metrics:\n"
                f"{fundamental_data_str}\n\n"
                "Respond with a JSON object like this: `{\"outlook\": \"Strong\"}`. "
                "The value for \"outlook\" should be one word: 'Strong', 'Moderate', or 'Weak'. No other text or filler."
            )
            self._update_llm_log_display(f"--- Company Fundamental Outlook ---\nPrompt:\n{fundamental_prompt}\n")
            fundamental_outlook_classification = "Moderate"  # Default
            fundamental_strength = 0.0  # Default strength

            try:
                fundamental_response = ollama.chat(
                    model='deepseek-r1', messages=[{'role': 'user', 'content': fundamental_prompt}], stream=False,
                    options={'num_predict': 20}, format='json'
                )
                fundamental_raw_answer = fundamental_response['message']['content'].strip()

                parsed_outlook = "Moderate"
                try:
                    json_data = json.loads(fundamental_raw_answer)
                    if 'outlook' in json_data:
                        parsed_outlook = json_data['outlook'].strip().capitalize()
                except json.JSONDecodeError:
                    self._update_llm_log_display(
                        f"Fundamental Outlook: JSON parse failed. Raw: {fundamental_raw_answer}. Falling back to regex.\n")
                    if re.search(r"strong", fundamental_raw_answer, re.IGNORECASE):
                        parsed_outlook = "Strong"
                    elif re.search(r"weak", fundamental_raw_answer, re.IGNORECASE):
                        parsed_outlook = "Weak"
                    elif re.search(r"moderate", fundamental_raw_answer, re.IGNORECASE):
                        parsed_outlook = "Moderate"

                fundamental_outlook_classification = parsed_outlook

                # If a specific outlook (Strong/Weak) is given, ask for strength
                if fundamental_outlook_classification in ["Strong", "Weak"]:
                    strength_prompt = (
                        f"On a scale of 0.0 to 1.0 (0.0=no strength, 1.0=very strong), "
                        f"how strong is this '{fundamental_outlook_classification}' fundamental outlook for {company_long_name} ({ticker})? "
                        "Respond with a JSON object like this: `{\"strength\": 0.8}`. No other text or filler."
                    )
                    self._update_llm_log_display(f"--- Company Fundamental Strength ---\nPrompt:\n{strength_prompt}\n")
                    strength_response = ollama.chat(
                        model='deepseek-r1', messages=[{'role': 'user', 'content': strength_prompt}], stream=False,
                        options={'num_predict': 20}, format='json'
                    )
                    strength_raw_answer = strength_response['message']['content'].strip()
                    self._update_llm_log_display(f"Raw Response: {strength_raw_answer}\n")
                    try:
                        json_data = json.loads(strength_raw_answer)
                        if 'strength' in json_data:
                            fundamental_strength = float(json_data['strength'])
                            fundamental_strength = max(0.0, min(1.0, fundamental_strength))  # Clamp between 0 and 1
                    except (json.JSONDecodeError, ValueError):
                        self._update_llm_log_display(
                            f"Fundamental Strength: JSON parse failed. Raw: {strength_raw_answer}. Falling back to regex.\n")
                        score_match = re.search(r"([-+]?\d*\.?\d+)", strength_raw_answer)
                        if score_match:
                            fundamental_strength = max(0.0, min(1.0, float(score_match.group(0))))

                self._update_llm_log_display(
                    f"Parsed Value: '{fundamental_outlook_classification}', Strength: {fundamental_strength:.2f}\n")

            except Exception as e:
                self._update_log_display(
                    f"Warning: DeepSeek-R1 fundamental outlook/strength failed: {e}. Defaulting to Moderate with 0 strength.",
                    clear_previous=False)
                self._update_llm_log_display(f"Error: {e}. Defaulting to Moderate with 0 strength.\n")
                fundamental_outlook_classification = "Moderate"
                fundamental_strength = 0.0

            self.cached_company_fundamental_outlook = fundamental_outlook_classification  # Cache for display
            fundamental_bias_magnitude = fundamental_strength * self.MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT
            self._update_log_display(
                f"DeepSeek-R1 Company Fundamental Outlook: {fundamental_outlook_classification} (Bias Magnitude: {fundamental_bias_magnitude * 100:.2f}%)\n",
                clear_previous=False)

            # --- Retrieve Recent News and Perform Sentiment Analysis (with ENHANCED DeepSeek-R1 relevance filtering and risk identification) ---
            self._update_status(
                f"Retrieving recent news for {ticker} and analyzing sentiment with {selected_analyzer}...", "blue")
            news_items = ticker_yf_obj.news

            relevant_news_items = []
            total_news_checked = len(news_items)
            self._update_log_display(
                f"Found {total_news_checked} total recent news items. Filtering with DeepSeek-R1 for relevance...",
                clear_previous=False)
            self._update_llm_log_display(f"--- News Relevance, Sentiment, and Risk Analysis ---\n")

            for i, item in enumerate(news_items):
                title = item.get('title') or item.get('headline') or 'No Title Provided'
                summary = item.get('content', {}).get('summary') or item.get('content', {}).get(
                    'description') or 'No Summary Provided'
                full_text_for_llm = f"Title: {title}\nSummary: {summary}".strip()

                if not full_text_for_llm:
                    continue

                relevance_prompt = (
                    f"Is this news article directly and primarily about {company_long_name} ({ticker})? "
                    "Respond with a JSON object like this: `{\"relevance\": \"YES\"}`. "
                    "The value for \"relevance\" should be one word: 'YES' or 'NO'. No other text or filler."
                    f"\n\nNews: {full_text_for_llm}"
                )
                self._update_llm_log_display(f"\nNews Item {i + 1} - Relevance Prompt:\n{relevance_prompt}\n")

                is_relevant = False
                try:
                    relevance_response = ollama.chat(
                        model='deepseek-r1', messages=[{'role': 'user', 'content': relevance_prompt}], stream=False,
                        options={'num_predict': 20}, format='json'
                    )
                    relevance_raw_answer = relevance_response['message']['content'].strip()

                    parsed_relevance = "NO"
                    try:
                        json_data = json.loads(relevance_raw_answer)
                        if 'relevance' in json_data:
                            parsed_relevance = json_data['relevance'].strip().upper()
                    except json.JSONDecodeError:
                        self._update_llm_log_display(
                            f"Relevance: JSON parse failed. Raw: {relevance_raw_answer}. Falling back to regex.\n")
                        if re.search(r"yes", relevance_raw_answer, re.IGNORECASE):
                            parsed_relevance = "YES"
                        elif re.search(r"no", relevance_raw_answer, re.IGNORECASE):
                            parsed_relevance = "NO"

                    self._update_llm_log_display(
                        f"Raw Response: {relevance_raw_answer}\nParsed Value: '{parsed_relevance}'\n")
                    if parsed_relevance == "YES":
                        is_relevant = True
                except Exception as e:
                    self._update_log_display(
                        f"Warning: DeepSeek-R1 relevance check failed for news item {i + 1}: {e}. Falling back to keyword match.",
                        clear_previous=False)
                    self._update_llm_log_display(f"Relevance Check Error: {e}. Falling back to keyword match.\n")
                    fallback_keywords = [ticker.lower()]
                    if company_long_name and company_long_name != ticker:
                        fallback_keywords.append(company_long_name.lower().split(' ')[0])
                        fallback_keywords.append(company_long_name.lower())

                    lower_full_text = full_text_for_llm.lower()
                    for keyword in set(fallback_keywords):
                        if keyword in lower_full_text:
                            is_relevant = True
                            break
                    if is_relevant:
                        self._update_llm_log_display(f"Fallback: News item {i + 1} included due to keyword match.\n")

                if is_relevant:
                    relevant_news_items.append(item)

            overall_latest_sentiment_score = 0.0
            individual_news_sentiment_data = []
            all_identified_risks = set()
            total_risk_severity_score = 0.0

            valid_sentiment_count = 0

            self._update_log_display(
                f"After DeepSeek-R1 relevance filtering, found {len(relevant_news_items)} relevant news items.",
                clear_previous=False)
            if not relevant_news_items:
                self._update_log_display(
                    "No relevant news found to analyze sentiment or risks. This will impact prediction accuracy.",
                    clear_previous=False)

            for i, item in enumerate(relevant_news_items):
                title = item.get('title') or item.get('headline') or 'No Title Provided'
                summary = item.get('content', {}).get('summary') or item.get('content', {}).get(
                    'description') or 'No Summary Provided'
                full_text = f"{title}. {summary}".strip()

                current_item_sentiment = 0.0
                deepseek_summary = ""  # New: To store DeepSeek-R1 generated summary

                if selected_analyzer == "DeepSeek-R1":
                    self._update_status(
                        f"DeepSeek-R1: Analyzing relevant news item {i + 1}/{len(relevant_news_items)} for sentiment & summary...",
                        "blue")
                    ollama_prompt = (
                        f"Analyze this news headline and summary for sentiment towards the company ({ticker}). "
                        "Respond with a JSON object like this: `{\"score\": 0.75, \"summary\": \"Company X had good earnings.\"}`. "
                        "The value for \"score\" should be a numerical sentiment score between -1.0 (very negative) and +1.0 (very positive). "
                        "The value for \"summary\" should be a brief, actionable summary (max 2-3 sentences). No other text or filler."
                        f"\n\nNews: {full_text}"
                    )
                    self._update_llm_log_display(f"\nNews Item {i + 1} - Sentiment/Summary Prompt:\n{ollama_prompt}\n")
                    try:
                        ollama_response = ollama.chat(
                            model='deepseek-r1', messages=[{'role': 'user', 'content': ollama_prompt}], stream=False,
                            options={'num_predict': 100}, format='json'
                        )
                        llm_raw_output = ollama_response['message']['content'].strip()
                        self._update_llm_log_display(f"Sentiment/Summary Raw Response: {llm_raw_output}\n")

                        parsed_score = 0.0
                        parsed_summary = ""
                        try:
                            json_data = json.loads(llm_raw_output)
                            if 'score' in json_data:
                                parsed_score = float(json_data['score'])
                            if 'summary' in json_data:
                                parsed_summary = json_data['summary'].strip()
                        except (json.JSONDecodeError, ValueError):
                            self._update_llm_log_display(
                                f"Sentiment/Summary: JSON parse failed. Raw: {llm_raw_output}. Falling back to regex.\n")
                            score_match = re.search(r"\"score\":\s*([-+]?\d*\.?\d+)", llm_raw_output)
                            if score_match:
                                parsed_score = float(score_match.group(1))
                            else:
                                if "positive" in llm_raw_output.lower():
                                    parsed_score = 0.7
                                elif "negative" in llm_raw_output.lower():
                                    parsed_score = -0.7
                                self._update_llm_log_display(
                                    f"Sentiment Fallback: Score not directly parsable. Guessing based on text.\n")

                            summary_match = re.search(r"\"summary\":\s*\"([^\"]*)\"", llm_raw_output)
                            if summary_match:
                                parsed_summary = summary_match.group(1).strip()
                            else:
                                self._update_llm_log_display(
                                    f"Summary Fallback: Could not parse summary from raw output.\n")

                        current_item_sentiment = parsed_score
                        deepseek_summary = parsed_summary
                        self._update_llm_log_display(
                            f"Parsed Score: {current_item_sentiment}, Summary: {deepseek_summary}\n")

                    except Exception as ollama_e:
                        current_item_sentiment = 0.0
                        deepseek_summary = "Error generating summary."
                        self._update_log_display(
                            f"Warning: DeepSeek-R1 sentiment/summary failed for item {i + 1}: {ollama_e}",
                            clear_previous=False)
                        self._update_llm_log_display(f"Sentiment/Summary Error: {ollama_e}.\n")

                elif selected_analyzer == "VADER":
                    sentiment_scores = self.vader_analyzer.polarity_scores(full_text)
                    current_item_sentiment = sentiment_scores['compound']
                    deepseek_summary = "N/A (VADER used)"

                elif selected_analyzer == "TextBlob":
                    analysis = TextBlob(full_text)
                    current_item_sentiment = analysis.sentiment.polarity
                    deepseek_summary = "N/A (TextBlob used)"

                # --- DeepSeek-R1 for Risk Factor Identification and Severity ---
                identified_risks_for_item = []
                current_risk_severity = 0.0  # Severity for this specific news item

                # MODIFIED RISK PROMPT: Broadened to include uncertainty and negative outlook
                risk_prompt = (
                    f"Does this news article about {company_long_name} ({ticker}) indicate a significant negative risk factor (e.g., Regulatory issue, Major lawsuit, Supply chain disruption, Strong competition threat, Product recall, Executive scandal, Negative analyst downgrade, **earnings uncertainty, negative outlook, potential for adverse market reaction**)? "
                    "Respond with a JSON object like this: `{\"risks\": [\"Regulatory issue\", \"Earnings uncertainty\"]}`. "
                    "The value for \"risks\" should be an array of strings, or an empty array if no significant risks are found. No other text or filler."
                    f"\n\nNews: {full_text}"
                )
                self._update_llm_log_display(f"News Item {i + 1} - Risk Identification Prompt:\n{risk_prompt}\n")
                try:
                    risk_response = ollama.chat(
                        model='deepseek-r1', messages=[{'role': 'user', 'content': risk_prompt}], stream=False,
                        options={'num_predict': 50}, format='json'
                    )
                    risk_raw_answer = risk_response['message']['content'].strip()
                    self._update_llm_log_display(f"Risk Identification Raw Response: {risk_raw_answer}\n")

                    parsed_risks_list = []
                    try:
                        json_data = json.loads(risk_raw_answer)
                        if 'risks' in json_data and isinstance(json_data['risks'], list):
                            parsed_risks_list = [str(r).strip() for r in json_data['risks'] if str(r).strip()]
                    except json.JSONDecodeError:
                        self._update_llm_log_display(
                            f"Risk Identification: JSON parse failed. Raw: {risk_raw_answer}. Falling back to regex.\n")
                        potential_risks = [r.strip() for r in re.split(r'[,;]', risk_raw_answer) if
                                           r.strip() and not re.search(r"^(ok|let|i need|think)", r.strip(),
                                                                       re.IGNORECASE)]
                        parsed_risks_list = [r for r in potential_risks if r.lower() != 'none']

                    identified_risks_for_item.extend(parsed_risks_list)
                    all_identified_risks.update(parsed_risks_list)
                    self._update_llm_log_display(
                        f"Parsed Risks: {', '.join(identified_risks_for_item) if identified_risks_for_item else 'None'}\n")

                    if identified_risks_for_item:  # Only ask for severity if risks were actually identified
                        severity_prompt = (
                            f"On a scale of 1 to 5 (1=very minor, 3=moderate, 5=very severe), "
                            f"how severe is the impact of the following identified risk(s) '{', '.join(identified_risks_for_item)}' for {company_long_name} ({ticker}) based on the news text provided? "
                            "Respond with a JSON object like this: `{\"severity\": 3.5}`. "
                            "The value for \"severity\" should be a single numerical score. No other text or filler."
                            f"\n\nNews: {full_text}"
                        )
                        self._update_llm_log_display(f"News Item {i + 1} - Risk Severity Prompt:\n{severity_prompt}\n")
                        self._update_status(f"DeepSeek-R1: Quantifying risk severity for item {i + 1}...", "blue")
                        severity_response = ollama.chat(
                            model='deepseek-r1', messages=[{'role': 'user', 'content': severity_prompt}], stream=False,
                            options={'num_predict': 20}, format='json'
                        )
                        severity_raw_answer = severity_response['message']['content'].strip()
                        self._update_llm_log_display(f"Risk Severity Raw Response: {severity_raw_answer}\n")

                        parsed_severity = 0.0
                        try:
                            json_data = json.loads(severity_raw_answer)
                            if 'severity' in json_data:
                                parsed_severity = float(json_data['severity'])
                        except (json.JSONDecodeError, ValueError):
                            self._update_llm_log_display(
                                f"Risk Severity: JSON parse failed. Raw: {severity_raw_answer}. Falling back to regex.\n")
                            score_match = re.search(r"([-+]?\d*\.?\d+)", severity_raw_answer)
                            if score_match:
                                parsed_severity = float(score_match.group(0))

                        try:
                            if not (1 <= parsed_severity <= 5):
                                parsed_severity = max(1, min(5, parsed_severity))
                            total_risk_severity_score += parsed_severity
                            self._update_llm_log_display(f"Parsed Severity: {parsed_severity}\n")
                        except ValueError:
                            self._update_log_display(
                                f"Warning: Could not convert '{parsed_severity}' to float for severity. Defaulting to 0.",
                                clear_previous=False)
                            self._update_llm_log_display(
                                f"Risk Severity Error: Float conversion failed. Defaulting to 0.\n")
                            current_risk_severity = 0.0

                        current_risk_severity = parsed_severity

                    else:
                        self._update_llm_log_display(
                            f"No significant risks identified, skipping severity quantification.\n")
                        current_risk_severity = 0.0
                except Exception as e:
                    self._update_log_display(
                        f"Warning: DeepSeek-R1 risk/severity analysis failed for item {i + 1}: {e}",
                        clear_previous=False)
                    self._update_llm_log_display(f"Risk/Severity Error: {e}.\n")

                individual_news_sentiment_data.append({
                    'title': title,
                    'summary': summary,
                    'deepseek_summary': deepseek_summary,  # Store DeepSeek-R1 generated summary
                    'sentiment': current_item_sentiment,
                    'analyzer': selected_analyzer,
                    'risks': identified_risks_for_item,
                    'risk_severity': current_risk_severity
                })
                overall_latest_sentiment_score += current_item_sentiment
                valid_sentiment_count += 1

            if valid_sentiment_count > 0:
                overall_latest_sentiment_score /= valid_sentiment_count
            else:
                overall_latest_sentiment_score = 0.0

            self.cached_news_sentiment_details = individual_news_sentiment_data
            self.cached_overall_sentiment = overall_latest_sentiment_score
            self.cached_risk_factors = list(all_identified_risks)

            # --- Generate Overall Market Insight Summary ---
            self._update_status("DeepSeek-R1: Generating overall market insight...", "blue")
            # Safely access title and summary using .get()
            all_news_titles_summaries = "\n".join(
                [f"Title: {item.get('title', 'N/A')}\nSummary: {item.get('summary', 'N/A')}" for item in
                 relevant_news_items])
            if not all_news_titles_summaries:
                all_news_titles_summaries = "No relevant news found."

            market_insight_prompt = (
                f"Based on the following news headlines and summaries for {company_long_name} ({ticker}), "
                f"and considering the overall sentiment score of {overall_latest_sentiment_score:.4f} and identified risks: {', '.join(self.cached_risk_factors) if self.cached_risk_factors else 'None'}, "
                "provide a concise, actionable 'Market Insight' summary (max 3-4 sentences) about the short-term outlook for this stock. "
                "Focus on key takeaways for an investor. Do not include any JSON or special formatting, just the plain text summary."
                f"\n\nNews Items:\n{all_news_titles_summaries}"
            )
            self._update_llm_log_display(f"\n--- Overall Market Insight Prompt ---\nPrompt:\n{market_insight_prompt}\n")
            try:
                market_insight_response = ollama.chat(
                    model='deepseek-r1', messages=[{'role': 'user', 'content': market_insight_prompt}], stream=False,
                    options={'num_predict': 200}
                )
                market_insight_raw_answer = market_insight_response['message']['content'].strip()
                self.cached_market_insight_summary = market_insight_raw_answer
                self._update_llm_log_display(f"Market Insight Raw Response:\n{market_insight_raw_answer}\n")
            except Exception as e:
                self.cached_market_insight_summary = "Failed to generate market insight due to LLM error."
                self._update_log_display(f"Warning: DeepSeek-R1 market insight generation failed: {e}",
                                         clear_previous=False)
                self._update_llm_log_display(f"Market Insight Error: {e}.\n")

            self._update_log_display(
                f"Overall Latest Sentiment Score (from relevant news): {overall_latest_sentiment_score:.4f}\n",
                clear_previous=False)
            self._update_log_display(
                f"Total Unique Risks Identified: {', '.join(self.cached_risk_factors) if self.cached_risk_factors else 'None'}\n",
                clear_previous=False)
            self._update_log_display(f"Overall Accumulated Risk Severity Score: {total_risk_severity_score:.2f}\n",
                                     clear_previous=False)
            self.master.after(0, self._update_news_details_display)

            # --- Step 3: Prepare Data for LSTM (Historical Prices with multiple features) ---
            self._update_status("Preparing historical data (with multiple features) for LSTM training...", "blue")

            data_for_lstm = ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume',
                                        'SMA_10', 'SMA_20', 'RSI', 'MACD', 'Signal_Line', 'MACD_Hist']].values

            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = feature_scaler.fit_transform(data_for_lstm)

            close_price_idx = ohlcv_data.columns.get_loc('Close')

            X, y = [], []
            if len(scaled_features) < lookback_window + 1:
                raise ValueError(
                    f"Not enough historical data ({len(scaled_features)} data points after cleaning) for lookback window of {lookback_window} days. Please extend date range.")

            for i in range(lookback_window, len(scaled_features)):
                X.append(scaled_features[i - lookback_window:i, :])
                y.append(scaled_features[i, close_price_idx])

            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], data_for_lstm.shape[1]))

            temp_close_data = data_for_lstm[:, close_price_idx].reshape(-1, 1)
            close_scaler = MinMaxScaler(feature_range=(0, 1))
            close_scaler.fit(temp_close_data)

            # --- Step 4: Train LSTM Model ---
            self._update_status("Training LSTM model...", "blue")
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stopping])
            self._update_status("LSTM model trained successfully!", "green")

            # --- Step 5: Predict Raw Future Prices (multi-step prediction) ---
            self._update_status(f"Predicting next {prediction_days} future trading days' raw stock prices...", "blue")

            current_input_sequence = scaled_features[-lookback_window:].reshape(1, lookback_window,
                                                                                scaled_features.shape[1])

            raw_predicted_scaled_prices = []
            raw_predicted_actual_prices = []

            for _ in range(prediction_days):  # Predict for `prediction_days`
                raw_predicted_scaled_price = model.predict(current_input_sequence, verbose=0)[0, 0]

                raw_predicted_actual_price = close_scaler.inverse_transform([[raw_predicted_scaled_price]])[0, 0]

                raw_predicted_scaled_prices.append(raw_predicted_scaled_price)
                raw_predicted_actual_prices.append(raw_predicted_actual_price)

                next_day_features = current_input_sequence[0, -1, :].copy()
                next_day_features[close_price_idx] = raw_predicted_scaled_price

                current_input_sequence = np.concatenate((current_input_sequence[:, 1:, :],
                                                         next_day_features.reshape(1, 1, scaled_features.shape[1])),
                                                        axis=1)

            # --- Step 6: Apply Sentiment/Trend/Macro/Fundamentals/Risk Adjustments & Confidence Intervals ---
            last_actual_price = ohlcv_data['Close'].iloc[-1]

            predicted_dates_ts = []
            current_date_for_prediction = pd.to_datetime(ohlcv_data.index.max())  # Ensure it's a pandas Timestamp

            for _ in range(prediction_days):
                current_date_for_prediction = current_date_for_prediction + pd.Timedelta(days=1)
                # Skip weekends
                while current_date_for_prediction.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    current_date_for_prediction = current_date_for_prediction + pd.Timedelta(days=1)
                predicted_dates_ts.append(current_date_for_prediction.to_pydatetime())  # Convert to Python datetime

            predicted_dates_for_display = predicted_dates_ts  # Already Python datetimes

            final_predicted_prices = []
            confidence_intervals = []  # Store [lower, upper] bounds
            influence_explanation_detail = []

            # Calculate historical volatility for confidence interval
            # Using 20-day rolling standard deviation of daily returns
            ohlcv_data['Daily_Return'] = ohlcv_data['Close'].pct_change()
            ohlcv_data['Volatility'] = ohlcv_data['Daily_Return'].rolling(window=20).std()
            # Handle cases where volatility might be NaN (e.g., not enough data for 20 days)
            avg_volatility = ohlcv_data['Volatility'].iloc[-1] if not ohlcv_data[
                'Volatility'].isnull().all() else 0.01  # Default if not enough data

            # Apply Macroeconomic Bias (dynamically scaled)
            macro_bias_effective = 0
            if self.cached_macro_outlook == "Bullish":
                macro_bias_effective = macro_strength * self.MAX_MACRO_ADJUSTMENT_PERCENT  # Use cached strength
            elif self.cached_macro_outlook == "Bearish":
                macro_bias_effective = -macro_strength * self.MAX_MACRO_ADJUSTMENT_PERCENT

            # Apply Company Fundamental Bias (dynamically scaled)
            fundamental_bias_effective = 0
            if self.cached_company_fundamental_outlook == "Strong":
                fundamental_bias_effective = fundamental_strength * self.MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT  # Use cached strength
            elif self.cached_company_fundamental_outlook == "Weak":
                fundamental_bias_effective = -fundamental_strength * self.MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT

            max_conceptual_total_severity = 5 * max(1, len(relevant_news_items))
            if max_conceptual_total_severity == 0: max_conceptual_total_severity = 1

            risk_adjustment_percentage_magnitude = self.MAX_RISK_ADJUSTMENT_PERCENT * (
                        total_risk_severity_score / max_conceptual_total_severity)
            risk_adjustment_value = -abs(risk_adjustment_percentage_magnitude)

            for idx, raw_predicted_price in enumerate(raw_predicted_actual_prices):
                adjusted_price = raw_predicted_price
                current_influence_msg = []

                if macro_bias_effective != 0:
                    adjusted_price += raw_predicted_price * macro_bias_effective
                    current_influence_msg.append(
                        f"Macro bias ({self.cached_macro_outlook} outlook, strength {macro_strength:.2f}): {macro_bias_effective * 100:.2f}%.")

                if fundamental_bias_effective != 0:
                    adjusted_price += raw_predicted_price * fundamental_bias_effective
                    current_influence_msg.append(
                        f"Company fundamentals ({self.cached_company_fundamental_outlook} outlook, strength {fundamental_strength:.2f}): {fundamental_bias_effective * 100:.2f}%.")

                sentiment_adjustment_effective = self.cached_overall_sentiment * self.MAX_SENTIMENT_ADJUSTMENT_PERCENT
                adjusted_price += raw_predicted_price * sentiment_adjustment_effective
                current_influence_msg.append(
                    f"News sentiment ({selected_analyzer} score: {self.cached_overall_sentiment:.4f}): {sentiment_adjustment_effective * 100:.2f}%.")

                if risk_adjustment_value != 0 and self.cached_risk_factors:
                    adjusted_price += raw_predicted_price * risk_adjustment_value
                    current_influence_msg.append(
                        f"Identified risks ({', '.join(self.cached_risk_factors)}) leading to a {abs(risk_adjustment_value * 100):.2f}% downward push (total severity {total_risk_severity_score:.1f}).")

                final_predicted_prices.append(max(0.01, adjusted_price))
                influence_explanation_detail.append(" ".join(current_influence_msg))

                # Calculate confidence interval (e.g., +/- 1.5 * avg_volatility * price for a wider band)
                # Multiplied by (idx + 1) to make the band wider for further predictions
                confidence_range = avg_volatility * adjusted_price * 1.5 * (idx + 1)
                confidence_intervals.append([adjusted_price - confidence_range, adjusted_price + confidence_range])

            # --- Final Display ---
            fundamental_display_str = ""
            if self.cached_fundamental_metrics_used:
                fundamental_display_items = [f"{k}: {v}" for k, v in self.cached_fundamental_metrics_used.items()]
                fundamental_display_str = f" ({', '.join(fundamental_display_items)})"

            prediction_text_lines = [
                f"Prediction for {ticker}:",
                f"Last Known Actual Close Price: ${last_actual_price:.2f}",
                f"DeepSeek-R1 Macroeconomic Outlook: {self.cached_macro_outlook} (Strength: {macro_strength:.2f})",
                f"DeepSeek-R1 Company Fundamental Outlook: {self.cached_company_fundamental_outlook} (Strength: {fundamental_strength:.2f}){fundamental_display_str}",
                f"Overall News Sentiment (via {selected_analyzer}): {self.cached_overall_sentiment:.4f}",
                f"Identified Specific Risks: {', '.join(self.cached_risk_factors) if self.cached_risk_factors else 'None'} (Total Severity: {total_risk_severity_score:.2f})",
                "\n--- Predicted Future Prices ---"
            ]

            for i in range(prediction_days):
                pred_date = predicted_dates_for_display[i].strftime('%Y-%m-%d')
                raw_pred = raw_predicted_actual_prices[i]
                final_pred = final_predicted_prices[i]
                conf_lower = confidence_intervals[i][0]
                conf_upper = confidence_intervals[i][1]
                influence_msg = influence_explanation_detail[i]

                prediction_text_lines.append(f"\nDay {i + 1} ({pred_date}):")
                prediction_text_lines.append(f"  Raw LSTM Predicted Price: ${raw_pred:.2f}")
                prediction_text_lines.append(f"  Influence Factors: {influence_msg}")
                prediction_text_lines.append(f"  Final Adjusted Predicted Price: ${final_pred:.2f}")
                prediction_text_lines.append(f"  Confidence Interval (95%): [${conf_lower:.2f}, ${conf_upper:.2f}]")

            final_prediction_text = "\n".join(prediction_text_lines)

            self.master.after(0, lambda: self.prediction_label.config(text=final_prediction_text, fg="darkgreen"))
            self.master.after(0, lambda: self._show_custom_messagebox("Prediction Complete",
                                                                      "Stock prices predicted with enhanced intelligence! Plots generated.",
                                                                      "success"))
            self._update_status("Prediction process complete! Plots generated. Click buttons to open.", "green")

            # --- Generate Plotly Plots ---
            self._update_status("Generating Plotly graphs and saving as HTML...", "blue")
            # Main prediction plot
            self._generate_and_save_plotly_html(ticker, ohlcv_data, predicted_dates_for_display, final_predicted_prices,
                                                confidence_intervals)
            # Technical indicators plot
            self._generate_and_save_indicators_plotly_html(ticker, ohlcv_data)

            self.master.after(0, lambda: self.open_plotly_button.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.open_indicators_button.config(state=tk.NORMAL))
            self._update_status("Plots generated! Click buttons to open.", "green")

        except Exception as caught_e:
            error_traceback = traceback.format_exc()
            error_message = f"An error occurred: {type(caught_e).__name__}: {caught_e}\n\nTraceback:\n{error_traceback}"

            self.master.after(0,
                              lambda: self._show_custom_messagebox("Error", f"An error occurred: {caught_e}", "error"))
            self.master.after(0, lambda msg=error_message: self._update_log_display(msg, clear_previous=False))
            self.master.after(0, lambda: self._update_status("Error during prediction. Check logs.", "red"))
            self.master.after(0, lambda e_val=caught_e: self.prediction_label.config(text=f"Prediction Failed: {e_val}",
                                                                                     fg="red"))
        finally:
            self.master.after(0, lambda: self.predict_button.config(state=tk.NORMAL))

    def _generate_and_save_plotly_html(self, ticker, historical_ohlcv_df, predicted_dates, predicted_prices,
                                       confidence_intervals):
        """
        Generates an interactive Plotly graph of historical (candlestick) and predicted prices
        with confidence intervals, and saves it as an HTML file.
        """
        fig = go.Figure()

        # Add Candlestick trace for historical data
        fig.add_trace(go.Candlestick(
            x=historical_ohlcv_df.index,
            open=historical_ohlcv_df['Open'],
            high=historical_ohlcv_df['High'],
            low=historical_ohlcv_df['Low'],
            close=historical_ohlcv_df['Close'],
            name='Historical Candlestick',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))

        # Add predicted future prices as a line plot
        last_hist_date = historical_ohlcv_df.index.max().to_pydatetime()
        last_hist_price = historical_ohlcv_df['Close'].iloc[-1]

        all_future_dates = [last_hist_date] + predicted_dates
        all_future_prices = [last_hist_price] + predicted_prices

        fig.add_trace(go.Scatter(
            x=all_future_dates,
            y=all_future_prices,
            mode='lines+markers',
            name='Predicted Future Prices',
            line=dict(color='blue', dash='dot', width=3),
            marker=dict(symbol='circle', size=8, color='blue')
        ))

        # Add confidence interval bands
        conf_x = [last_hist_date] + predicted_dates
        conf_upper = [last_hist_price] + [ci[1] for ci in confidence_intervals]
        conf_lower = [last_hist_price] + [ci[0] for ci in confidence_intervals]

        fig.add_trace(go.Scatter(
            x=conf_x + conf_x[::-1],  # x, then x reversed
            y=conf_upper + conf_lower[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',  # Blue transparent fill
            line=dict(color='rgba(255,255,255,0)'),  # Transparent line
            name='Confidence Interval',
            hoverinfo='skip'
        ))

        # Add individual labels for predicted points
        for i, (date, price) in enumerate(zip(predicted_dates, predicted_prices)):
            fig.add_annotation(
                x=date,
                y=price,
                text=f'Day {i + 1}: ${price:.2f}',
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color='blue', size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="blue",
                borderwidth=1,
                borderpad=4,
                xref="x",
                yref="y"
            )

        fig.update_layout(
            title=f'{ticker} Stock Price: Historical & Predicted (DeepSeek-R1 Enhanced)',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,  # Hide range slider for cleaner look
            hovermode='x unified',
            template='plotly_white',
            height=800,  # Increased height for better visualization
            width=1600  # Increased width for better visualization
        )

        pio.write_html(fig, file=self.plotly_html_file, auto_open=False)

    def _generate_and_save_indicators_plotly_html(self, ticker, historical_ohlcv_df):
        """
        Generates an interactive Plotly graph for technical indicators (SMA, RSI, MACD).
        """
        # Create subplots: 3 rows, 1 column, shared x-axis
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,  # Increased spacing
                            row_titles=['Price/SMA', 'RSI', 'MACD'])

        # Row 1: Price and SMAs
        fig.add_trace(
            go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['Close'], mode='lines', name='Close Price',
                       line=dict(color='blue', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['SMA_10'], mode='lines', name='SMA 10',
                       line=dict(color='orange', width=2)), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['SMA_20'], mode='lines', name='SMA 20',
                       line=dict(color='purple', width=2)), row=1, col=1)

        # Row 2: RSI
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['RSI'], mode='lines', name='RSI',
                                 line=dict(color='cyan', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Overbought",
                      annotation_position="top right", annotation_font_color="red")  # Overbought
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1, annotation_text="Oversold",
                      annotation_position="bottom right", annotation_font_color="green")  # Oversold

        # Row 3: MACD
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['MACD'], mode='lines', name='MACD',
                                 line=dict(color='gold', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['Signal_Line'], mode='lines',
                                 name='Signal Line', line=dict(color='magenta', width=2)), row=3, col=1)

        # MACD Histogram colors based on positive/negative values
        macd_hist_colors = ['rgba(0,255,0,0.5)' if val >= 0 else 'rgba(255,0,0,0.5)' for val in
                            historical_ohlcv_df['MACD_Hist']]
        fig.add_trace(go.Bar(x=historical_ohlcv_df.index, y=historical_ohlcv_df['MACD_Hist'], name='MACD Hist',
                             marker_color=macd_hist_colors), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

        fig.update_layout(
            title=f'{ticker} Technical Indicators',
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white',
            height=1000,  # Increased height for more vertical space
            width=1600  # Increased width for better visualization
        )

        pio.write_html(fig, file=self.plotly_indicators_html_file, auto_open=False)


# --- Main Application Run ---
if __name__ == "__main__":
    # IMPORTANT: Before running this script, ensure you have the following Python libraries installed:
    # pip install yfinance pandas numpy scikit-learn tensorflow vaderSentiment textblob plotly ollama

    # For TextBlob, you might need to download NLTK corpora:
    # python -m textblob.download_corpora

    # Also, ensure Ollama is installed and running on your system,
    # and you have pulled the 'deepseek-r1' model (e.g., by running 'ollama pull deepseek-r1' in your terminal).

    root = tk.Tk()
    app = ForwardLookingStockPredictor(root)
    root.mainloop()

