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

# Plotly imports
import plotly.graph_objects as go
import plotly.io as pio

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
    and DeepSeek-R1 interaction logs.
    It predicts two future trading days, skipping the immediate next trading day.
    Includes enhanced DeepSeek-R1 integration for news relevance filtering, dynamic prediction influence,
    macroeconomic outlook, automated company fundamental outlook, and specific risk factor identification.
    Graphs are generated using Plotly and opened in a web browser.
    DeepSeek-R1 responses are now requested in JSON format for dynamic adjustments, with robust fallback parsing.
    LSTM now incorporates more features (Open, High, Low, Volume, Technical Indicators).
    MAX_ADJUSTMENT_PERCENT values have been reduced to prevent over-correction by the LLM.
    """

    # Define max adjustment percentages for dynamic scaling. These are tunable.
    # Reduced values to prevent over-correction based on user feedback.
    MAX_MACRO_ADJUSTMENT_PERCENT = 0.002  # Max 0.2% adjustment based on macro strength
    MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT = 0.003  # Max 0.3% adjustment based on fundamental strength
    MAX_SENTIMENT_ADJUSTMENT_PERCENT = 0.005  # Max 0.5% adjustment based on sentiment score (-1 to +1)
    MAX_RISK_ADJUSTMENT_PERCENT = 0.010  # Max 1.0% downward adjustment based on total risk severity

    def __init__(self, master):
        self.master = master
        master.title("Forward-Looking Stock Predictor with Enhanced Intelligence")
        master.geometry("1300x880")  # Larger window for tabs and logs
        master.resizable(True, True)

        # Main Notebook (Tabbed Interface)
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 1: Prediction & Plot ---
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="Prediction & Plot")
        self.prediction_tab.grid_columnconfigure(0, weight=1)  # Left panel (inputs, logs)
        self.prediction_tab.grid_columnconfigure(1, weight=1)  # Right panel (plot frame, now with Plotly button)
        self.prediction_tab.grid_rowconfigure(0, weight=0)  # Input frame fixed size
        self.prediction_tab.grid_rowconfigure(1, weight=1)  # Logs and Plotly button area expand vertically

        # --- Input Parameters Frame (Tidied Up) ---
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

        tk.Label(self.input_frame, text="Neutral Trend Lookback Days:").grid(row=row_idx, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.neutral_trend_lookback_entry = tk.Entry(self.input_frame)
        self.neutral_trend_lookback_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        self.neutral_trend_lookback_entry.insert(0, "5")

        tk.Label(self.input_frame, text="Sentiment Analyzer:").grid(row=row_idx, column=2, padx=5, pady=5, sticky="w")
        self.analyzer_choice = ttk.Combobox(self.input_frame,
                                            values=["DeepSeek-R1", "VADER", "TextBlob"],
                                            state="readonly")
        self.analyzer_choice.set("DeepSeek-R1")
        self.analyzer_choice.grid(row=row_idx, column=3, padx=5, pady=5, sticky="ew")
        row_idx += 1

        self.predict_button = tk.Button(self.input_frame, text="Predict Prices & Generate Plot",
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

        # --- Right Panel: Plotly Button Frame ---
        self.plot_controls_frame = tk.LabelFrame(self.prediction_tab, text="View Plot", padx=5, pady=5)
        self.plot_controls_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.plot_controls_frame.grid_rowconfigure(0, weight=1)  # Spacer for vertical alignment
        self.plot_controls_frame.grid_rowconfigure(1, weight=0)  # Button row
        self.plot_controls_frame.grid_columnconfigure(0, weight=1)

        self.open_plotly_button = tk.Button(self.plot_controls_frame, text="Open Interactive Plotly Graph",
                                            command=self._open_plotly_html, state=tk.DISABLED)
        self.open_plotly_button.grid(row=1, column=0, padx=10, pady=10, sticky="s")  # Stick to bottom

        # --- Tab 2: News & Sentiment Details ---
        self.news_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.news_tab, text="Recent News & Sentiment Details")
        self.news_tab.grid_columnconfigure(0, weight=1)
        self.news_tab.grid_rowconfigure(0, weight=0)
        self.news_tab.grid_rowconfigure(1, weight=1)

        tk.Label(self.news_tab, text="--- Latest News Items and Individual Sentiment Scores ---",
                 font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.news_details_text = scrolledtext.ScrolledText(self.news_tab, wrap=tk.WORD, width=100, height=30,
                                                           state='disabled')
        self.news_details_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # --- Tab 3: DeepSeek-R1 Interactions ---
        self.llm_interactions_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.llm_interactions_tab, text="DeepSeek-R1 Interactions")
        self.llm_interactions_tab.grid_columnconfigure(0, weight=1)
        self.llm_interactions_tab.grid_rowconfigure(0, weight=1)

        self.llm_log_text = scrolledtext.ScrolledText(self.llm_interactions_tab, wrap=tk.WORD, state='disabled')
        self.llm_log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- Status Bar (at bottom of main window) ---
        self.status_label = tk.Label(master, text="Ready.", relief=tk.SUNKEN, bd=1, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()

        self.plotly_html_file = "stock_prediction_plot.html"  # Name for the Plotly output file

        # Initial Ollama status check in a separate thread
        self._start_ollama_status_check()

        # Caches for display
        self.cached_news_sentiment_details = []
        self.cached_overall_sentiment = 0.0
        self.cached_macro_outlook = "N/A"
        self.cached_company_fundamental_outlook = "N/A"
        self.cached_risk_factors = []
        self.cached_fundamental_metrics_used = {}
        self.cached_llm_interactions_log = ""  # New cache for LLM interaction logs

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
                    self.news_details_text.insert(tk.END, f"Title: {item['title']}\n")
                    self.news_details_text.insert(tk.END, f"Summary: {item['summary']}\n")
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
            messagebox.showerror("File Not Found", "Plotly HTML file not found. Please run a prediction first.")
            self._update_status("Plotly HTML file not found.", "red")

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
        neutral_trend_lookback_str = self.neutral_trend_lookback_entry.get()
        selected_analyzer = self.analyzer_choice.get()

        if not all([ticker, start_date_str, end_date_str, lookback_window_str, neutral_trend_lookback_str,
                    selected_analyzer]):
            messagebox.showwarning("Input Error", "Please fill in all required fields.")
            return

        try:
            lookback_window = int(lookback_window_str)
            neutral_trend_lookback_days = int(neutral_trend_lookback_str)
            datetime.strptime(start_date_str, '%Y-%m-%d')
            datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            messagebox.showwarning("Input Error",
                                   "Lookback window and neutral trend days must be integers. Dates must be YYYY-MM-DD.")
            return

        self.predict_button.config(state=tk.DISABLED)
        self.open_plotly_button.config(state=tk.DISABLED)  # Disable Plotly button until new plot is ready
        self._update_status("Starting prediction process...", "blue")
        self._update_log_display("", clear_previous=True)
        self._update_llm_log_display("", append=False)  # Clear LLM log tab
        self.prediction_label.config(text="Prediction: Processing...")
        self.cached_news_sentiment_details = []
        self.cached_overall_sentiment = 0.0
        self.cached_macro_outlook = "N/A"
        self.cached_company_fundamental_outlook = "N/A"
        self.cached_risk_factors = []
        self.cached_fundamental_metrics_used = {}
        self._update_news_details_display()

        thread = threading.Thread(target=self._perform_prediction_task,
                                  args=(ticker, start_date_str, end_date_str,
                                        lookback_window, neutral_trend_lookback_days,
                                        selected_analyzer))
        thread.daemon = True
        thread.start()

    def _perform_prediction_task(self, ticker, start_date_str, end_date_str, lookback_window,
                                 neutral_trend_lookback_days, selected_analyzer):
        """
        Main task to fetch data, analyze sentiment, train LSTM, and predict.
        Runs in a separate thread.
        """
        try:
            start_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_date_dt >= end_date_dt:
                raise ValueError("Start date must be before end date.")

            # --- Step 1: Retrieve Historical Stock Data (now with more features) ---
            self._update_status(
                f"Downloading historical stock data for {ticker} from {start_date_str} to {end_date_str}...", "blue")
            ticker_yf_obj = yf.Ticker(ticker)
            # Fetch Open, High, Low, Close, Volume
            stock_data = ticker_yf_obj.history(start=start_date_str, end=end_date_dt, auto_adjust=False)
            if stock_data.empty:
                raise ValueError(
                    f"No stock data found for {ticker} in the specified date range. Please check ticker or dates.")

            # Ensure column names are standard (e.g., 'Adj Close' to 'Close')
            if 'Adj Close' in stock_data.columns:
                stock_data['Close'] = stock_data['Adj Close']

            # Select relevant columns for LSTM and calculations
            # Use a list of columns that are expected
            ohlcv_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            ohlcv_data.index = pd.to_datetime(ohlcv_data.index).normalize()
            ohlcv_data.index.name = 'date'
            ohlcv_data.sort_index(inplace=True)

            self._update_log_display(f"Raw OHLCV data head:\n{ohlcv_data.head().to_string()}\n")

            # --- Calculate Technical Indicators ---
            self._update_status("Calculating technical indicators...", "blue")

            # Simple Moving Averages (SMA)
            ohlcv_data['SMA_10'] = ohlcv_data['Close'].rolling(window=10).mean()
            ohlcv_data['SMA_20'] = ohlcv_data['Close'].rolling(window=20).mean()

            # Relative Strength Index (RSI) - 14 periods
            # Calculate daily price changes
            delta = ohlcv_data['Close'].diff()
            # Separate gains (positive changes) and losses (negative changes, absolute value)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate Exponential Moving Average of gains and losses
            avg_gain = gain.ewm(span=14, adjust=False).mean()
            avg_loss = loss.ewm(span=14, adjust=False).mean()

            # Calculate Relative Strength (RS) and RSI
            # Handle division by zero for avg_loss by setting rs to a very large number if avg_loss is zero
            rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
            ohlcv_data['RSI'] = 100 - (100 / (1 + rs))
            # First 14 values of RSI will be NaN, then it will start calculating

            # MACD (Moving Average Convergence Divergence) - 12, 26, 9 periods
            exp1 = ohlcv_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = ohlcv_data['Close'].ewm(span=26, adjust=False).mean()
            ohlcv_data['MACD'] = exp1 - exp2
            ohlcv_data['Signal_Line'] = ohlcv_data['MACD'].ewm(span=9, adjust=False).mean()
            ohlcv_data['MACD_Hist'] = ohlcv_data['MACD'] - ohlcv_data['Signal_Line']

            # Drop rows with NaN values created by rolling/ewm calculations
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

            stock_price_df = ohlcv_data[
                ['Close']].copy()  # Keep only Close for plotting purposes later, but LSTM uses all.

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
            company_long_name = ticker_yf_obj.info.get('longName', ticker)

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
                if selected_analyzer == "DeepSeek-R1":
                    self._update_status(
                        f"DeepSeek-R1: Analyzing relevant news item {i + 1}/{len(relevant_news_items)} for sentiment...",
                        "blue")
                    ollama_prompt = (
                        f"Analyze this news headline and summary for sentiment towards the company ({ticker}). "
                        "Respond with a JSON object like this: `{\"score\": 0.75}`. "
                        "The value for \"score\" should be a numerical sentiment score between -1.0 (very negative) and +1.0 (very positive). No other text or filler."
                        f"\n\nNews: {full_text}"
                    )
                    self._update_llm_log_display(f"\nNews Item {i + 1} - Sentiment Prompt:\n{ollama_prompt}\n")
                    try:
                        ollama_response = ollama.chat(
                            model='deepseek-r1', messages=[{'role': 'user', 'content': ollama_prompt}], stream=False,
                            options={'num_predict': 20}, format='json'
                        )
                        llm_raw_output = ollama_response['message']['content'].strip()
                        self._update_llm_log_display(f"Sentiment Raw Response: {llm_raw_output}\n")

                        parsed_score = 0.0
                        try:
                            json_data = json.loads(llm_raw_output)
                            if 'score' in json_data:
                                parsed_score = float(json_data['score'])
                        except (json.JSONDecodeError, ValueError):
                            self._update_llm_log_display(
                                f"Sentiment: JSON parse failed. Raw: {llm_raw_output}. Falling back to regex.\n")
                            score_match = re.search(r"([-+]?\d*\.?\d+)", llm_raw_output)
                            if score_match:
                                parsed_score = float(score_match.group(0))
                            else:
                                if "positive" in llm_raw_output.lower():
                                    parsed_score = 0.7
                                elif "negative" in llm_raw_output.lower():
                                    parsed_score = -0.7
                                self._update_llm_log_display(
                                    f"Sentiment Fallback: Score not directly parsable. Guessing based on text.\n")

                        current_item_sentiment = parsed_score
                        self._update_llm_log_display(f"Parsed Score: {current_item_sentiment}\n")

                    except Exception as ollama_e:
                        current_item_sentiment = 0.0
                        self._update_log_display(
                            f"Warning: DeepSeek-R1 sentiment analysis failed for item {i + 1}: {ollama_e}",
                            clear_previous=False)
                        self._update_llm_log_display(f"Sentiment Error: {ollama_e}.\n")

                elif selected_analyzer == "VADER":
                    sentiment_scores = self.vader_analyzer.polarity_scores(full_text)
                    current_item_sentiment = sentiment_scores['compound']

                elif selected_analyzer == "TextBlob":
                    analysis = TextBlob(full_text)
                    current_item_sentiment = analysis.sentiment.polarity

                # --- DeepSeek-R1 for Risk Factor Identification and Severity ---
                identified_risks_for_item = []
                current_risk_severity = 0.0  # Severity for this specific news item

                risk_prompt = (
                    f"Does this news article about {company_long_name} ({ticker}) indicate a significant negative risk factor (e.g., Regulatory issue, Major lawsuit, Supply chain disruption, Strong competition threat, Product recall, Executive scandal, Negative analyst downgrade)? "
                    "Respond with a JSON object like this: `{\"risks\": [\"Regulatory issue\", \"Major lawsuit\"]}`. "
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

            # Select features for LSTM input. Ensure 'Close' is always present for scaling/inverse.
            # Make sure all columns are numeric
            data_for_lstm = ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume',
                                        'SMA_10', 'SMA_20', 'RSI', 'MACD', 'Signal_Line', 'MACD_Hist']].values

            # Scale all features
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = feature_scaler.fit_transform(data_for_lstm)

            # Separate target variable (Close price) from scaled features for LSTM training
            # The target (y) is still the 'Close' price, so we need its column index
            close_price_idx = ohlcv_data.columns.get_loc('Close')

            X, y = [], []
            # lookback_window must be less than or equal to the number of available data points
            if len(scaled_features) < lookback_window + 1:
                raise ValueError(
                    f"Not enough historical data ({len(scaled_features)} data points after cleaning) for lookback window of {lookback_window} days. Please extend date range.")

            for i in range(lookback_window, len(scaled_features)):
                X.append(scaled_features[i - lookback_window:i, :])  # All features for lookback window
                y.append(scaled_features[i, close_price_idx])  # Only the scaled Close price for prediction

            X, y = np.array(X), np.array(y)

            # Reshape X for LSTM input: (samples, timesteps, features)
            # The number of features is now data_for_lstm.shape[1]
            X = np.reshape(X, (X.shape[0], X.shape[1], data_for_lstm.shape[1]))

            # Separate scaler for inverse transforming the predicted 'Close' price
            # This scaler only works on the 'Close' price column's min/max from the original training data.
            # We will create a dummy scaler that only scales/inverses the 'Close' price column
            temp_close_data = data_for_lstm[:, close_price_idx].reshape(-1, 1)
            close_scaler = MinMaxScaler(feature_range=(0, 1))
            close_scaler.fit(temp_close_data)  # Fit only on the Close price column

            # --- Step 4: Train LSTM Model ---
            self._update_status("Training LSTM model...", "blue")
            model = Sequential()
            # Input shape adjusted to (lookback_window, number_of_features)
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))  # Output is still a single price

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stopping])
            self._update_status("LSTM model trained successfully!", "green")

            # --- Step 5: Predict Raw Future Prices (2-day multi-step prediction) ---
            self._update_status("Predicting next 2 future trading days' raw stock prices...", "blue")

            # Get the last 'lookback_window' of all features for the initial prediction input
            current_input_sequence = scaled_features[-lookback_window:].reshape(1, lookback_window,
                                                                                scaled_features.shape[1])

            raw_predicted_scaled_prices = []
            raw_predicted_actual_prices = []

            for i in range(2):  # Predict for 2 future days
                raw_predicted_scaled_price = model.predict(current_input_sequence, verbose=0)[0, 0]

                # Inverse transform only the predicted Close price using the dedicated close_scaler
                raw_predicted_actual_price = close_scaler.inverse_transform([[raw_predicted_scaled_price]])[0, 0]

                raw_predicted_scaled_prices.append(raw_predicted_scaled_price)
                raw_predicted_actual_prices.append(raw_predicted_actual_price)

                # Update the input sequence for the next prediction
                # For simplicity, for the subsequent predictions, we only append the predicted Close price
                # and assume other features (Open, High, Low, Volume, Indicators) for the next day
                # are not available or are estimated to remain stable relative to the Close.
                # In a more advanced setup, you might predict/estimate other features too.
                # Here, we'll replace the last 'Close' price in the sequence with the new prediction,
                # and leave other feature values as they were in the last actual timestep.
                # This is a simplification; a truly robust multi-feature prediction would need to
                # predict all features for the next step, or use more sophisticated imputation.

                # Create a placeholder row for the predicted day's scaled features
                # Copy the last actual scaled features row
                next_day_features = current_input_sequence[0, -1, :].copy()
                # Update the 'Close' price in this placeholder row with the new prediction
                next_day_features[close_price_idx] = raw_predicted_scaled_price

                # Append this placeholder to the current input sequence
                current_input_sequence = np.concatenate((current_input_sequence[:, 1:, :],
                                                         next_day_features.reshape(1, 1, scaled_features.shape[1])),
                                                        axis=1)

            # --- Step 6: Apply Sentiment/Trend/Macro/Fundamentals/Risk Adjustments to Each Prediction ---
            last_actual_price = ohlcv_data['Close'].iloc[-1]

            predicted_dates_ts = []
            current_date_ts = ohlcv_data.index.max()

            current_date_ts += timedelta(days=1)
            while current_date_ts.weekday() >= 5:  # Skip weekends
                current_date_ts += timedelta(days=1)

            for _ in range(2):  # Predict for 2 future trading days
                current_date_ts += timedelta(days=1)
                while current_date_ts.weekday() >= 5:  # Skip weekends
                    current_date_ts += timedelta(days=1)
                predicted_dates_ts.append(current_date_ts)

            predicted_dates_for_display = [dt.to_pydatetime() for dt in predicted_dates_ts]

            final_predicted_prices = []
            influence_explanation_detail = []  # Collect influence messages

            # Apply Macroeconomic Bias (dynamically scaled)
            macro_bias_effective = 0
            if macro_outlook_classification == "Bullish":
                macro_bias_effective = macro_bias_magnitude
            elif macro_outlook_classification == "Bearish":
                macro_bias_effective = -macro_bias_magnitude
            # Neutral remains 0

            # Apply Company Fundamental Bias (dynamically scaled)
            fundamental_bias_effective = 0
            if fundamental_outlook_classification == "Strong":
                fundamental_bias_effective = fundamental_bias_magnitude
            elif fundamental_outlook_classification == "Weak":
                fundamental_bias_effective = -fundamental_bias_magnitude
            # Moderate remains 0

            # Calculate effective risk adjustment based on total_risk_severity_score
            max_conceptual_total_severity = 5 * max(1, len(relevant_news_items))
            if max_conceptual_total_severity == 0: max_conceptual_total_severity = 1  # Avoid division by zero

            risk_adjustment_percentage_magnitude = self.MAX_RISK_ADJUSTMENT_PERCENT * (
                        total_risk_severity_score / max_conceptual_total_severity)
            risk_adjustment_value = -abs(risk_adjustment_percentage_magnitude)  # Always negative or zero

            for raw_predicted_price in raw_predicted_actual_prices:
                adjusted_price = raw_predicted_price
                current_influence_msg = []

                # 1. Apply Macroeconomic Bias
                if macro_bias_effective != 0:
                    adjusted_price += raw_predicted_price * macro_bias_effective
                    current_influence_msg.append(
                        f"Macro bias ({macro_outlook_classification} outlook, strength {macro_strength:.2f}): {macro_bias_effective * 100:.2f}%.")

                # 2. Apply Company Fundamental Bias
                if fundamental_bias_effective != 0:
                    adjusted_price += raw_predicted_price * fundamental_bias_effective
                    current_influence_msg.append(
                        f"Company fundamentals ({fundamental_outlook_classification} outlook, strength {fundamental_strength:.2f}): {fundamental_bias_effective * 100:.2f}%.")

                # 3. Apply Sentiment/Trend based on Company News (dynamically scaled by sentiment score)
                # Note: overall_latest_sentiment_score is already -1.0 to +1.0
                sentiment_adjustment_effective = overall_latest_sentiment_score * self.MAX_SENTIMENT_ADJUSTMENT_PERCENT
                adjusted_price += raw_predicted_price * sentiment_adjustment_effective
                current_influence_msg.append(
                    f"News sentiment ({selected_analyzer} score: {overall_latest_sentiment_score:.4f}): {sentiment_adjustment_effective * 100:.2f}%.")

                # 4. Apply Dynamic Risk Adjustment (if any risks identified)
                if risk_adjustment_value != 0 and self.cached_risk_factors:
                    adjusted_price += raw_predicted_price * risk_adjustment_value
                    current_influence_msg.append(
                        f"Identified risks ({', '.join(self.cached_risk_factors)}) leading to a {abs(risk_adjustment_value * 100):.2f}% downward push (total severity {total_risk_severity_score:.1f}).")

                final_predicted_prices.append(
                    max(0.01, adjusted_price))  # Ensure price doesn't go below negligible value
                influence_explanation_detail.append(" ".join(current_influence_msg))

            # --- Final Display ---
            # Prepare fundamental metrics string for display
            fundamental_display_str = ""
            if self.cached_fundamental_metrics_used:
                fundamental_display_items = [f"{k}: {v}" for k, v in self.cached_fundamental_metrics_used.items()]
                fundamental_display_str = f" ({', '.join(fundamental_display_items)})"

            prediction_text = (
                f"Prediction for {ticker}:\n"
                f"Last Known Actual Close Price: ${last_actual_price:.2f}\n"
                f"Raw LSTM Predicted Day 1 ({predicted_dates_for_display[0].strftime('%Y-%m-%d')}): ${raw_predicted_actual_prices[0]:.2f}\n"
                f"Raw LSTM Predicted Day 2 ({predicted_dates_for_display[1].strftime('%Y-%m-%d')}): ${raw_predicted_actual_prices[1]:.2f}\n"
                f"DeepSeek-R1 Macroeconomic Outlook: {macro_outlook_classification} (Strength: {macro_strength:.2f})\n"
                f"DeepSeek-R1 Company Fundamental Outlook: {fundamental_outlook_classification} (Strength: {fundamental_strength:.2f}){fundamental_display_str}\n"
                f"Overall News Sentiment (via {selected_analyzer}): {overall_latest_sentiment_score:.4f}\n"
                f"Identified Specific Risks: {', '.join(self.cached_risk_factors) if self.cached_risk_factors else 'None'} (Total Severity: {total_risk_severity_score:.2f})\n"
                f"\nInfluence on Day 1 Prediction: {influence_explanation_detail[0]}\n"
                f"Final Adjusted Predicted Day 1 Price: ${final_predicted_prices[0]:.2f}\n\n"
                f"Influence on Day 2 Prediction: {influence_explanation_detail[1]}\n"
                f"Final Adjusted Predicted Day 2 Price: ${final_predicted_prices[1]:.2f}"
            )
            self.master.after(0, lambda: self.prediction_label.config(text=prediction_text, fg="darkgreen"))
            self.master.after(0, lambda: messagebox.showinfo("Prediction Complete",
                                                             "Stock prices predicted with enhanced intelligence! Plotly graph generated and ready to open."))
            self._update_status(
                "Prediction process complete! Plotly graph generated. Click 'Open Interactive Plotly Graph' button.",
                "green")

            # --- Generate Plotly Plot and Save as HTML ---
            self._update_status("Generating Plotly graph and saving as HTML...", "blue")
            self._generate_and_save_plotly_html(ticker, stock_price_df, predicted_dates_for_display,
                                                final_predicted_prices)
            self.master.after(0, lambda: self.open_plotly_button.config(state=tk.NORMAL))
            self._update_status("Plotly HTML generated! Click 'Open Interactive Plotly Graph' button.", "green")

        except Exception as caught_e:
            error_traceback = traceback.format_exc()
            error_message = f"An error occurred: {type(caught_e).__name__}: {caught_e}\n\nTraceback:\n{error_traceback}"

            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {caught_e}"))
            self.master.after(0, lambda msg=error_message: self._update_log_display(msg, clear_previous=False))
            self.master.after(0, lambda: self._update_status("Error during prediction. Check logs.", "red"))
            self.master.after(0, lambda e_val=caught_e: self.prediction_label.config(text=f"Prediction Failed: {e_val}",
                                                                                     fg="red"))
        finally:
            self.master.after(0, lambda: self.predict_button.config(state=tk.NORMAL))

    def _generate_and_save_plotly_html(self, ticker, historical_df, predicted_dates, predicted_prices):
        """
        Generates an interactive Plotly graph of historical and predicted prices and saves it as an HTML file.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df['Close'],
            mode='lines',
            name='Historical Close Price',
            line=dict(color='blue')
        ))

        all_future_dates = [historical_df.index.max().to_pydatetime()] + predicted_dates
        all_future_prices = [historical_df['Close'].iloc[-1]] + predicted_prices

        fig.add_trace(go.Scatter(
            x=all_future_dates,
            y=all_future_prices,
            mode='lines+markers',
            name='Predicted Future Prices',
            line=dict(color='red', dash='dot'),
            marker=dict(symbol='circle', size=8, color='red')
        ))

        for i, (date, price) in enumerate(zip(predicted_dates, predicted_prices)):
            fig.add_annotation(
                x=date,
                y=price,
                text=f'Day {i + 1}: ${price:.2f}',
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                font=dict(color='red', size=12),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="red",
                borderwidth=1,
                borderpad=4,
                xref="x",
                yref="y"
            )

        fig.update_layout(
            title=f'{ticker} Stock Price: Historical & Predicted (DeepSeek-R1 Enhanced)',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=900
        )

        pio.write_html(fig, file=self.plotly_html_file, auto_open=False)


# --- Main Application Run ---
if __name__ == "__main__":
    # IMPORTANT: Before running this script, ensure you have the following Python libraries installed:
    # pip install yfinance pandas numpy scikit-learn tensorflow vaderSentiment textblob plotly

    # For TextBlob, you might need to download NLTK corpora:
    # python -m textblob.download_corpora

    # Also, ensure Ollama is installed and running on your system,
    # and you have pulled the 'deepseek-r1' model (e.g., by running 'ollama pull deepseek-r1' in your terminal).

    root = tk.Tk()
    app = ForwardLookingStockPredictor(root)
    root.mainloop()