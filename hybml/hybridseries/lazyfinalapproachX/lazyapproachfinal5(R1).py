import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import yfinance as yf
import pandas as pd
import numpy as np
import threading
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import traceback

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
    The application features a tabbed UI for prediction results and detailed news sentiment.
    It predicts two future trading days, skipping the immediate next trading day.
    Includes enhanced DeepSeek-R1 integration for news relevance filtering, dynamic prediction influence,
    macroeconomic outlook, automated company fundamental outlook, and specific risk factor identification.
    """

    def __init__(self, master):
        self.master = master
        master.title("Forward-Looking Stock Predictor with Sentiment")
        master.geometry("1200x850")  # Larger window for plot and info
        master.resizable(True, True)

        # Main Notebook (Tabbed Interface) for primary layout
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 1: Prediction & Plot ---
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="Prediction & Plot")
        # Configure grid for this tab: inputs at top, logs/plot below
        self.prediction_tab.grid_columnconfigure(0, weight=1)  # Left panel (inputs, logs)
        self.prediction_tab.grid_columnconfigure(1, weight=1)  # Right panel (plot)
        self.prediction_tab.grid_rowconfigure(0, weight=0)  # Input frame fixed size
        self.prediction_tab.grid_rowconfigure(1, weight=1)  # Logs and plot expand vertically

        # --- Input Parameters Frame ---
        self.input_frame = tk.LabelFrame(self.prediction_tab, text="Prediction Parameters", padx=10, pady=10)
        self.input_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        # Configure columns within input_frame for labels and entries
        self.input_frame.grid_columnconfigure(1, weight=1)  # Entry column expands

        current_row = 0
        tk.Label(self.input_frame, text="Company Ticker (e.g., AAPL):").grid(row=current_row, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.ticker_entry = tk.Entry(self.input_frame)
        self.ticker_entry.grid(row=current_row, column=1, padx=5, pady=5, sticky="ew")
        current_row += 1

        tk.Label(self.input_frame, text="Historical Data Start Date (YYYY-MM-DD):").grid(row=current_row, column=0,
                                                                                         padx=5, pady=5, sticky="w")
        self.start_date_entry = tk.Entry(self.input_frame)
        self.start_date_entry.grid(row=current_row, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365 * 2)).strftime(
            '%Y-%m-%d'))  # Default 2 years back
        current_row += 1

        tk.Label(self.input_frame, text="Historical Data End Date (YYYY-MM-DD):").grid(row=current_row, column=0,
                                                                                       padx=5, pady=5, sticky="w")
        self.end_date_entry = tk.Entry(self.input_frame)
        self.end_date_entry.grid(row=current_row, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, (datetime.now() - timedelta(days=1)).strftime(
            '%Y-%m-%d'))  # End date is typically yesterday
        current_row += 1

        tk.Label(self.input_frame, text="LSTM Lookback Window (days):").grid(row=current_row, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.lookback_entry = tk.Entry(self.input_frame)
        self.lookback_entry.grid(row=current_row, column=1, padx=5, pady=5, sticky="ew")
        self.lookback_entry.insert(0, "60")  # Default lookback window for LSTM
        current_row += 1

        tk.Label(self.input_frame, text="Neutral Trend Lookback Days:").grid(row=current_row, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.neutral_trend_lookback_entry = tk.Entry(self.input_frame)
        self.neutral_trend_lookback_entry.grid(row=current_row, column=1, padx=5, pady=5, sticky="ew")
        self.neutral_trend_lookback_entry.insert(0, "5")  # Default lookback for neutral trend
        current_row += 1

        # Removed manual revenue_growth and profit_margin inputs

        tk.Label(self.input_frame, text="Choose Sentiment Analyzer:").grid(row=current_row, column=0, padx=5, pady=5,
                                                                           sticky="w")
        self.analyzer_choice = ttk.Combobox(self.input_frame,
                                            values=["DeepSeek-R1", "VADER", "TextBlob"],  # Updated options
                                            state="readonly")
        self.analyzer_choice.set("DeepSeek-R1")  # Default choice to DeepSeek-R1
        self.analyzer_choice.grid(row=current_row, column=1, padx=5, pady=5, sticky="ew")
        current_row += 1

        self.predict_button = tk.Button(self.input_frame, text="Predict Prices & Plot",
                                        command=self._start_prediction_thread)
        self.predict_button.grid(row=current_row, column=0, columnspan=2, pady=10, sticky="ew")
        current_row += 1

        self.ollama_status_label = tk.Label(self.input_frame, text="Ollama Status: Checking...", fg="gray")
        self.ollama_status_label.grid(row=current_row, column=0, columnspan=2, pady=5, sticky="ew")

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

        # --- Right Panel: Plotting Frame ---
        self.plot_frame = tk.LabelFrame(self.prediction_tab, text="Historical Data & Predicted Price", padx=5, pady=5)
        self.plot_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)  # Plot canvas expands
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.toolbar_frame = tk.Frame(self.plot_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # --- Tab 2: News & Sentiment Details ---
        self.news_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.news_tab, text="Recent News & Sentiment Details")
        self.news_tab.grid_columnconfigure(0, weight=1)  # News content expands
        self.news_tab.grid_rowconfigure(0, weight=0)  # Title fixed
        self.news_tab.grid_rowconfigure(1, weight=1)  # ScrolledText expands

        tk.Label(self.news_tab, text="--- Latest News Items and Individual Sentiment Scores ---",
                 font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.news_details_text = scrolledtext.ScrolledText(self.news_tab, wrap=tk.WORD, width=100, height=30,
                                                           state='disabled')
        self.news_details_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # --- Status Bar (at bottom of main window) ---
        self.status_label = tk.Label(master, text="Ready.", relief=tk.SUNKEN, bd=1, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Initial Ollama status check in a separate thread
        self._start_ollama_status_check()

        # Store news and sentiment for display in the news tab
        self.cached_news_sentiment_details = []
        self.cached_overall_sentiment = 0.0
        self.cached_macro_outlook = "N/A"
        self.cached_company_fundamental_outlook = "N/A"
        self.cached_risk_factors = []
        self.cached_fundamental_metrics_used = {}  # New cache for fundamental metrics used

    def _update_status(self, message, color="black"):
        """Helper to update the status bar label."""
        self.status_label.config(text=message, fg=color)
        self.master.update_idletasks()

    def _update_log_display(self, text_to_add, clear_previous=True):
        """Helper to safely update the scrolledtext widget for logs."""
        self.log_text_display.config(state='normal')
        if clear_previous:
            self.log_text_display.delete(1.0, tk.END)
        self.log_text_display.insert(tk.END, text_to_add + "\n")
        self.log_text_display.see(tk.END)  # Scroll to the end
        self.log_text_display.config(state='disabled')

    def _update_news_details_display(self):
        """Helper to update the scrolledtext widget for news details."""
        self.news_details_text.config(state='normal')
        self.news_details_text.delete(1.0, tk.END)

        if not self.cached_news_sentiment_details and not self.cached_risk_factors:
            self.news_details_text.insert(tk.END, "No recent news found or analyzed for the last prediction run.\n")
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
                self.news_details_text.insert(tk.END, "No relevant news found to display details.\n")

        self.news_details_text.see(tk.END)
        self.news_details_text.config(state='disabled')

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
        self._update_status("Starting prediction process...", "blue")
        self._update_log_display("", clear_previous=True)
        self.prediction_label.config(text="Prediction: Processing...")
        self.ax.clear()
        self.canvas.draw_idle()
        self.cached_news_sentiment_details = []
        self.cached_overall_sentiment = 0.0
        self.cached_macro_outlook = "N/A"
        self.cached_company_fundamental_outlook = "N/A"
        self.cached_risk_factors = []
        self.cached_fundamental_metrics_used = {}  # Clear cached fundamental metrics
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

            # --- Step 1: Retrieve Historical Stock Data ---
            self._update_status(
                f"Downloading historical stock data for {ticker} from {start_date_str} to {end_date_str}...", "blue")
            ticker_yf_obj = yf.Ticker(ticker)  # Get Ticker object here to use its .info and .news
            stock_data = ticker_yf_obj.history(start=start_date_str, end=end_date_dt, auto_adjust=False)
            if stock_data.empty:
                raise ValueError(
                    f"No stock data found for {ticker} in the specified date range. Please check ticker or dates.")

            flattened_columns = []
            for col in stock_data.columns:
                if isinstance(col, tuple):
                    flattened_columns.append(col[0])
                else:
                    flattened_columns.append(col)
            stock_data.columns = flattened_columns

            stock_price_df = None
            if 'Close' in stock_data.columns:
                stock_price_df = stock_data[['Close']].copy()
            elif 'Adj Close' in stock_data.columns:
                stock_price_df = stock_data[['Adj Close']].copy()
                stock_price_df.columns = ['Close']
            else:
                raise KeyError(
                    f"Could not find 'Close' or 'Adj Close' column after processing for {ticker}. Available columns: {stock_data.columns.tolist()}")

            stock_price_df.index = pd.to_datetime(stock_price_df.index).normalize()
            stock_price_df.index.name = 'date'
            stock_price_df.sort_index(inplace=True)

            self._update_log_display(f"Stock data head:\n{stock_price_df.head().to_string()}\n")
            self._update_log_display(f"Stock data tail:\n{stock_price_df.tail().to_string()}\n", clear_previous=False)
            self._update_log_display(f"Stock data index type: {type(stock_price_df.index)}\n", clear_previous=False)
            self._update_log_display(f"Stock data columns: {stock_price_df.columns.tolist()}\n", clear_previous=False)

            # --- ENHANCEMENT 1: DeepSeek-R1 for Macroeconomic Outlook ---
            self._update_status("DeepSeek-R1: Assessing overall macroeconomic outlook...", "blue")
            macro_prompt = (
                "Based on general global economic conditions and recent news trends (e.g., inflation, interest rates, GDP, geopolitical events), "
                "how would you classify the current short-term (next 1-2 weeks) macroeconomic outlook for the stock market? "
                "Respond strictly with one word: 'Bullish', 'Neutral', or 'Bearish'."
            )
            macro_outlook_classification = "Neutral"  # Default
            try:
                macro_response = ollama.chat(
                    model='deepseek-r1', messages=[{'role': 'user', 'content': macro_prompt}], stream=False,
                    options={'num_predict': 10}
                )
                macro_answer = macro_response['message']['content'].strip().upper()
                if "BULLISH" in macro_answer:
                    macro_outlook_classification = "Bullish"
                elif "BEARISH" in macro_answer:
                    macro_outlook_classification = "Bearish"
                else:
                    macro_outlook_classification = "Neutral"
            except Exception as e:
                self._update_log_display(f"Warning: DeepSeek-R1 macro outlook failed: {e}. Defaulting to Neutral.",
                                         clear_previous=False)
                macro_outlook_classification = "Neutral"

            self.cached_macro_outlook = macro_outlook_classification  # Cache for display
            self._update_log_display(f"DeepSeek-R1 Macroeconomic Outlook: {macro_outlook_classification}\n",
                                     clear_previous=False)

            # --- ENHANCEMENT 2: DeepSeek-R1 for Automated Company Fundamental Outlook ---
            self._update_status(f"DeepSeek-R1: Assessing {ticker} company fundamental outlook...", "blue")
            company_long_name = ticker_yf_obj.info.get('longName', ticker)

            # Extract fundamental metrics from ticker_yf_obj.info
            company_info = ticker_yf_obj.info
            fundamental_metrics = {}

            # Common profitability/growth metrics
            if 'revenuePerShare' in company_info and company_info['revenuePerShare'] is not None:
                fundamental_metrics['Revenue Per Share'] = company_info['revenuePerShare']
            if 'profitMargins' in company_info and company_info['profitMargins'] is not None:
                fundamental_metrics[
                    'Profit Margins'] = f"{company_info['profitMargins'] * 100:.2f}%"  # Convert to percentage
            elif 'grossMargins' in company_info and company_info['grossMargins'] is not None:
                fundamental_metrics['Gross Margins'] = f"{company_info['grossMargins'] * 100:.2f}%"
            elif 'operatingMargins' in company_info and company_info['operatingMargins'] is not None:
                fundamental_metrics['Operating Margins'] = f"{company_info['operatingMargins'] * 100:.2f}%"

            # Valuation & Efficiency metrics
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

            # Construct a descriptive string for DeepSeek-R1
            fundamental_data_str = "No specific fundamental data found via yfinance.info."
            if fundamental_metrics:
                fundamental_data_str = ", ".join([f"{k}: {v}" for k, v in fundamental_metrics.items()])

            self.cached_fundamental_metrics_used = fundamental_metrics  # Cache for display

            fundamental_prompt = (
                f"Considering the following fundamental metrics for {company_long_name} ({ticker}):\n"
                f"{fundamental_data_str}\n\n"
                f"Based on these metrics and general financial knowledge of the company's industry, "
                f"how would you classify its current fundamental health and outlook for investors? "
                "Respond strictly with one word: 'Strong', 'Moderate', or 'Weak'."
            )
            fundamental_outlook_classification = "Moderate"  # Default
            try:
                fundamental_response = ollama.chat(
                    model='deepseek-r1', messages=[{'role': 'user', 'content': fundamental_prompt}], stream=False,
                    options={'num_predict': 10}
                )
                fundamental_answer = fundamental_response['message']['content'].strip().upper()
                if "STRONG" in fundamental_answer:
                    fundamental_outlook_classification = "Strong"
                elif "WEAK" in fundamental_answer:
                    fundamental_outlook_classification = "Weak"
                else:
                    fundamental_outlook_classification = "Moderate"
            except Exception as e:
                self._update_log_display(
                    f"Warning: DeepSeek-R1 fundamental outlook failed: {e}. Defaulting to Moderate.",
                    clear_previous=False)
                fundamental_outlook_classification = "Moderate"

            self.cached_company_fundamental_outlook = fundamental_outlook_classification  # Cache for display
            self._update_log_display(f"DeepSeek-R1 Company Fundamental Outlook: {fundamental_outlook_classification}\n",
                                     clear_previous=False)

            # --- Step 2: Retrieve Recent News and Perform Sentiment Analysis (with ENHANCED DeepSeek-R1 relevance filtering and risk identification) ---
            self._update_status(
                f"Retrieving recent news for {ticker} and analyzing sentiment with {selected_analyzer}...", "blue")
            news_items = ticker_yf_obj.news  # Use the already obtained ticker_yf_obj

            relevant_news_items = []
            total_news_checked = len(news_items)
            self._update_log_display(
                f"Found {total_news_checked} total recent news items. Filtering with DeepSeek-R1 for relevance...",
                clear_previous=False)

            for i, item in enumerate(news_items):
                title = item.get('title') or item.get('headline') or 'No Title Provided'
                summary = item.get('content', {}).get('summary') or item.get('content', {}).get(
                    'description') or 'No Summary Provided'
                full_text_for_llm = f"Title: {title}\nSummary: {summary}".strip()

                if not full_text_for_llm:
                    continue

                relevance_prompt = (
                    f"Is this news article directly and primarily about {company_long_name} ({ticker})? "
                    "Respond strictly with 'YES' or 'NO'.\n\n"
                    f"News: {full_text_for_llm}"
                )
                try:
                    relevance_response = ollama.chat(
                        model='deepseek-r1', messages=[{'role': 'user', 'content': relevance_prompt}], stream=False,
                        options={'num_predict': 10}
                    )
                    relevance_answer = relevance_response['message']['content'].strip().upper()
                    if "YES" in relevance_answer:
                        relevant_news_items.append(item)
                except Exception as e:
                    self._update_log_display(
                        f"Warning: DeepSeek-R1 relevance check failed for news item {i + 1}: {e}. Falling back to keyword match.",
                        clear_previous=False)
                    fallback_keywords = [ticker.lower()]
                    if company_long_name and company_long_name != ticker:
                        fallback_keywords.append(company_long_name.lower().split(' ')[0])
                        fallback_keywords.append(company_long_name.lower())

                    is_relevant_by_fallback = False
                    lower_full_text = full_text_for_llm.lower()
                    for keyword in set(fallback_keywords):
                        if keyword in lower_full_text:
                            is_relevant_by_fallback = True
                            break
                    if is_relevant_by_fallback:
                        relevant_news_items.append(item)
                        self._update_log_display(f"Fallback: News item {i + 1} included due to keyword match.",
                                                 clear_previous=False)

            overall_latest_sentiment_score = 0.0
            individual_news_sentiment_data = []
            all_identified_risks = set()  # Use a set to store unique risks
            total_risk_severity_score = 0.0  # Accumulate severity

            valid_sentiment_count = 0

            self._update_log_display(
                f"After DeepSeek-R1 relevance filtering, found {len(relevant_news_items)} relevant news items.",
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
                        "Provide a numerical sentiment score between -1.0 (very negative) and +1.0 (very positive). "
                        "Format your response strictly as: Score: [SCORE]\n\n"
                        f"News: {full_text}"
                    )
                    try:
                        ollama_response = ollama.chat(
                            model='deepseek-r1', messages=[{'role': 'user', 'content': ollama_prompt}], stream=False
                        )
                        llm_output = ollama_response['message']['content'].strip()
                        score_match = re.search(r"Score: ([-+]?\d*\.?\d+)", llm_output)
                        if score_match:
                            current_item_sentiment = float(score_match.group(1))
                        else:
                            current_item_sentiment = 0.0
                            if "positive" in llm_output.lower():
                                current_item_sentiment = 0.7
                            elif "negative" in llm_output.lower():
                                current_item_sentiment = -0.7

                    except Exception as ollama_e:
                        current_item_sentiment = 0.0
                        self._update_log_display(
                            f"Warning: DeepSeek-R1 sentiment analysis failed for item {i + 1}: {ollama_e}",
                            clear_previous=False)

                elif selected_analyzer == "VADER":
                    sentiment_scores = self.vader_analyzer.polarity_scores(full_text)
                    current_item_sentiment = sentiment_scores['compound']

                elif selected_analyzer == "TextBlob":
                    analysis = TextBlob(full_text)
                    current_item_sentiment = analysis.sentiment.polarity

                # --- ENHANCEMENT 3: DeepSeek-R1 for Risk Factor Identification and Severity ---
                identified_risks_for_item = []
                current_risk_severity = 0.0  # Severity for this specific news item

                risk_prompt = (
                    f"Does this news article about {company_long_name} ({ticker}) indicate a significant negative risk factor (e.g., Regulatory issue, Major lawsuit, Supply chain disruption, Strong competition threat, Product recall, Executive scandal, Negative analyst downgrade)? "
                    "List any identified risks, separated by commas. If no significant risks, respond with 'None'.\n\n"
                    f"News: {full_text}"
                )
                try:
                    self._update_status(
                        f"DeepSeek-R1: Checking news item {i + 1}/{len(relevant_news_items)} for risks...", "blue")
                    risk_response = ollama.chat(
                        model='deepseek-r1', messages=[{'role': 'user', 'content': risk_prompt}], stream=False,
                        options={'num_predict': 50}
                    )
                    risk_answer = risk_response['message']['content'].strip()
                    if risk_answer.lower() != 'none' and risk_answer != '':
                        risks = [r.strip() for r in re.split(r'[,;]', risk_answer) if r.strip()]
                        identified_risks_for_item.extend(risks)
                        all_identified_risks.update(risks)  # Add to overall set of risks

                        # Now, quantify severity for *this specific risk*
                        severity_prompt = (
                            f"On a scale of 1 to 5 (1=very minor, 3=moderate, 5=very severe), "
                            f"how severe is the impact of the following identified risk(s) '{', '.join(risks)}' for {company_long_name} ({ticker}) based on the news text provided? "
                            "Respond strictly with a single numerical score.\n\n"
                            f"News: {full_text}"
                        )
                        self._update_status(f"DeepSeek-R1: Quantifying risk severity for item {i + 1}...", "blue")
                        severity_response = ollama.chat(
                            model='deepseek-r1', messages=[{'role': 'user', 'content': severity_prompt}], stream=False,
                            options={'num_predict': 5}
                        )
                        severity_answer = severity_response['message']['content'].strip()
                        try:
                            current_risk_severity = float(severity_answer)
                            if not (1 <= current_risk_severity <= 5):  # Clamp to valid range
                                current_risk_severity = max(1, min(5, current_risk_severity))
                            total_risk_severity_score += current_risk_severity  # Accumulate
                        except ValueError:
                            self._update_log_display(
                                f"Warning: DeepSeek-R1 severity score for item {i + 1} was not a number: '{severity_answer}'. Defaulting to 0.",
                                clear_previous=False)
                            current_risk_severity = 0.0  # Default if parsing fails
                except Exception as e:
                    self._update_log_display(
                        f"Warning: DeepSeek-R1 risk/severity analysis failed for item {i + 1}: {e}",
                        clear_previous=False)

                individual_news_sentiment_data.append({
                    'title': title,
                    'summary': summary,
                    'sentiment': current_item_sentiment,
                    'analyzer': selected_analyzer,
                    'risks': identified_risks_for_item,
                    'risk_severity': current_risk_severity  # Store severity for individual news display
                })
                overall_latest_sentiment_score += current_item_sentiment
                valid_sentiment_count += 1

            if valid_sentiment_count > 0:
                overall_latest_sentiment_score /= valid_sentiment_count
            else:
                overall_latest_sentiment_score = 0.0

            self.cached_news_sentiment_details = individual_news_sentiment_data
            self.cached_overall_sentiment = overall_latest_sentiment_score
            self.cached_risk_factors = list(all_identified_risks)  # Convert set to list for caching

            self._update_log_display(
                f"Overall Latest Sentiment Score (from relevant news): {overall_latest_sentiment_score:.4f}\n",
                clear_previous=False)
            self._update_log_display(
                f"Total Unique Risks Identified: {', '.join(self.cached_risk_factors) if self.cached_risk_factors else 'None'}\n",
                clear_previous=False)
            self._update_log_display(f"Overall Accumulated Risk Severity Score: {total_risk_severity_score:.2f}\n",
                                     clear_previous=False)
            self.master.after(0, self._update_news_details_display)

            # --- Step 3: Prepare Data for LSTM (Historical Prices Only) ---
            self._update_status("Preparing historical data for LSTM training...", "blue")

            data_for_lstm = stock_price_df['Close'].values.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_for_lstm)

            X, y = [], []
            if len(scaled_data) < lookback_window + 1:
                raise ValueError(
                    f"Not enough historical data ({len(scaled_data)} data points) for lookback window of {lookback_window} days. Please extend date range.")

            for i in range(lookback_window, len(scaled_data)):
                X.append(scaled_data[i - lookback_window:i, 0])
                y.append(scaled_data[i, 0])

            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # --- Step 4: Train LSTM Model ---
            self._update_status("Training LSTM model...", "blue")
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stopping])
            self._update_status("LSTM model trained successfully!", "green")

            # --- Step 5: Predict Raw Future Prices (2-day multi-step prediction) ---
            self._update_status("Predicting next 2 future trading days' raw stock prices...", "blue")

            current_input_sequence = scaled_data[-lookback_window:].reshape(1, lookback_window, 1)

            raw_predicted_scaled_prices = []
            raw_predicted_actual_prices = []

            for i in range(2):  # Predict for 2 future days
                raw_predicted_scaled_price = model.predict(current_input_sequence, verbose=0)[0, 0]
                raw_predicted_actual_price = scaler.inverse_transform([[raw_predicted_scaled_price]])[0, 0]

                raw_predicted_scaled_prices.append(raw_predicted_scaled_price)
                raw_predicted_actual_prices.append(raw_predicted_actual_price)

                current_input_sequence = np.concatenate((current_input_sequence[:, 1:, :],
                                                         raw_predicted_scaled_price.reshape(1, 1, 1)),
                                                        axis=1)

            # --- Step 6: Apply Sentiment/Trend/Macro/Fundamentals/Risk Adjustments to Each Prediction ---
            last_actual_price = stock_price_df['Close'].iloc[-1]

            predicted_dates_ts = []
            current_date_ts = stock_price_df.index.max()

            current_date_ts += timedelta(days=1)
            while current_date_ts.weekday() >= 5:
                current_date_ts += timedelta(days=1)

            for _ in range(2):
                current_date_ts += timedelta(days=1)
                while current_date_ts.weekday() >= 5:
                    current_date_ts += timedelta(days=1)
                predicted_dates_ts.append(current_date_ts)

            predicted_dates_for_display = [dt.to_pydatetime() for dt in predicted_dates_ts]

            final_predicted_prices = []
            influence_explanation_detail = []  # Collect influence messages

            # Define adjustment factors
            sentiment_positive_threshold = 0.2
            sentiment_negative_threshold = -0.2
            sentiment_neutral_min_threshold = -0.1
            sentiment_neutral_max_threshold = 0.1

            base_sentiment_adjustment_percentage = 0.008

            macro_bias_values = {
                "Bullish": 0.003,  # +0.3% global bias
                "Neutral": 0,
                "Bearish": -0.003  # -0.3% global bias
            }
            macro_bias = macro_bias_values.get(macro_outlook_classification, 0)

            fundamental_bias_values = {
                "Strong": 0.005,  # +0.5% for strong fundamentals
                "Moderate": 0,
                "Weak": -0.005  # -0.5% for weak fundamentals
            }
            fundamental_bias = fundamental_bias_values.get(fundamental_outlook_classification, 0)

            # Dynamic risk adjustment based on total_risk_severity_score
            # Max possible severity from a single news item is 5.
            # Scale total_risk_severity_score (which can be 0 to 5 * num_relevant_news)
            # to a percentage adjustment. Let's cap max downward adjustment at 3% for total severity.
            # Assume 10 relevant news items * 5 max severity = 50 max total severity
            # This can be tuned.
            max_conceptual_total_severity = 5 * max(1, len(relevant_news_items))  # Scale based on how much news we got

            risk_adjustment_percentage_magnitude = 0.03 * (
                        total_risk_severity_score / max_conceptual_total_severity)  # Max 3% downward
            risk_adjustment_value = -abs(risk_adjustment_percentage_magnitude)  # Always negative

            for raw_predicted_price in raw_predicted_actual_prices:
                adjusted_price = raw_predicted_price
                current_influence_msg = []

                # 1. Apply Macroeconomic Bias
                if macro_bias != 0:
                    adjusted_price += raw_predicted_price * macro_bias
                    current_influence_msg.append(
                        f"Macro bias ({macro_outlook_classification} outlook): {macro_bias * 100:.1f}%.")

                # 2. Apply Company Fundamental Bias
                if fundamental_bias != 0:
                    adjusted_price += raw_predicted_price * fundamental_bias
                    current_influence_msg.append(
                        f"Company fundamentals ({fundamental_outlook_classification} outlook): {fundamental_bias * 100:.1f}%.")

                # 3. Apply Sentiment/Trend based on Company News
                if sentiment_neutral_min_threshold <= overall_latest_sentiment_score <= sentiment_neutral_max_threshold:
                    if len(stock_price_df) < neutral_trend_lookback_days:
                        current_influence_msg.append(
                            f"Neutral news ({selected_analyzer} score: {overall_latest_sentiment_score:.2f}). No base trend data. Relying on LSTM trend.")
                    else:
                        recent_prices = stock_price_df['Close'].tail(neutral_trend_lookback_days)
                        if len(recent_prices) > 1:
                            daily_returns = recent_prices.pct_change().dropna()
                            if not daily_returns.empty:
                                avg_daily_return = daily_returns.mean()
                                base_trend_adjustment_amount = raw_predicted_price * avg_daily_return
                                adjusted_price += base_trend_adjustment_amount
                                current_influence_msg.append(
                                    f"Neutral news ({selected_analyzer} score: {overall_latest_sentiment_score:.2f}). "
                                    f"Adjusted by base trend ({avg_daily_return:.2%}) from last {neutral_trend_lookback_days} days."
                                )
                            else:
                                current_influence_msg.append(
                                    f"Neutral news ({selected_analyzer} score: {overall_latest_sentiment_score:.2f}). No valid daily returns for base trend. Relying on LSTM trend.")
                        else:
                            current_influence_msg.append(
                                f"Neutral news ({selected_analyzer} score: {overall_latest_sentiment_score:.2f}). Not enough data for base trend. Relying on LSTM trend.")
                elif overall_latest_sentiment_score > sentiment_positive_threshold:
                    current_influence_msg.append(
                        f"Strong Positive news ({selected_analyzer} score: {overall_latest_sentiment_score:.2f}). Adjusting upwards by {base_sentiment_adjustment_percentage * 100:.1f}%.")
                    adjusted_price += abs(raw_predicted_price * base_sentiment_adjustment_percentage)
                elif overall_latest_sentiment_score < sentiment_negative_threshold:
                    current_influence_msg.append(
                        f"Strong Negative news ({selected_analyzer} score: {overall_latest_sentiment_score:.2f}). Adjusting downwards by {base_sentiment_adjustment_percentage * 100:.1f}%.")
                    adjusted_price -= abs(raw_predicted_price * base_sentiment_adjustment_percentage)
                else:
                    current_influence_msg.append(
                        f"Mild news ({selected_analyzer} score: {overall_latest_sentiment_score:.2f}). Relying primarily on LSTM trend.")

                # 4. Apply Dynamic Risk Adjustment (if any risks identified)
                if risk_adjustment_value != 0 and self.cached_risk_factors:
                    adjusted_price += raw_predicted_price * risk_adjustment_value
                    current_influence_msg.append(
                        f"Identified risks ({', '.join(self.cached_risk_factors)}) leading to a {abs(risk_adjustment_value * 100):.1f}% downward push (total severity {total_risk_severity_score:.1f}).")

                final_predicted_prices.append(max(0.01, adjusted_price))
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
                f"Overall News Sentiment (via {selected_analyzer}): {overall_latest_sentiment_score:.4f}\n"
                f"DeepSeek-R1 Macroeconomic Outlook: {macro_outlook_classification}\n"
                f"DeepSeek-R1 Company Fundamental Outlook: {fundamental_outlook_classification}{fundamental_display_str}\n"  # Updated display
                f"Identified Specific Risks: {', '.join(self.cached_risk_factors) if self.cached_risk_factors else 'None'}\n"
                f"\nInfluence on Day 1 Prediction: {influence_explanation_detail[0]}\n"
                f"Final Adjusted Predicted Day 1 Price: ${final_predicted_prices[0]:.2f}\n\n"
                f"Influence on Day 2 Prediction: {influence_explanation_detail[1]}\n"
                f"Final Adjusted Predicted Day 2 Price: ${final_predicted_prices[1]:.2f}"
            )
            self.master.after(0, lambda: self.prediction_label.config(text=prediction_text, fg="darkgreen"))
            self.master.after(0, lambda: messagebox.showinfo("Prediction Complete",
                                                             "Stock prices predicted with enhanced intelligence!"))
            self._update_status("Prediction process complete!", "green")

            # --- Step 7: Plotting Historical Data and Future Prediction ---
            self._update_status("Generating plot of historical data and future prediction...", "blue")
            self._plot_prediction_and_history(ticker, stock_price_df, predicted_dates_for_display,
                                              final_predicted_prices)
            self._update_status("Plot generated! Check the right panel.", "green")

        except Exception as caught_e:
            error_traceback = traceback.format_exc()
            error_message = f"An error occurred: {type(caught_e).__name__}: {caught_e}\n\nTraceback:\n{error_traceback}"

            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {caught_e}"))
            self.master.after(0, lambda msg=error_message: self._update_log_display(msg, clear_previous=False))
            self.master.after(0, lambda: self._update_status("Error during prediction. Check logs.", "red"))
        finally:
            self.master.after(0, lambda: self.predict_button.config(state=tk.NORMAL))

    def _plot_prediction_and_history(self, ticker, historical_df, predicted_dates, predicted_prices):
        """
        Plots the historical stock prices and extends the graph with the two
        sentiment-adjusted future predicted prices.
        """
        self.ax.clear()

        self.ax.plot(historical_df.index, historical_df['Close'], label='Historical Close Price', color='blue')

        last_historical_date = historical_df.index.max().to_pydatetime()

        all_future_dates = [last_historical_date] + predicted_dates
        all_future_prices = [historical_df['Close'].iloc[-1]] + predicted_prices

        self.ax.plot(all_future_dates, all_future_prices, 'ro--', label='Predicted Future Prices', markersize=8)

        for i, (date, price) in enumerate(zip(predicted_dates, predicted_prices)):
            self.ax.annotate(f'Day {i + 1}: ${price:.2f}', (date, price), textcoords="offset points", xytext=(0, 10),
                             ha='center', color='red')

        self.ax.set_title(f'{ticker} Stock Price: Historical & Predicted (Sentiment-Adjusted)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.autofmt_xdate()
        self.canvas.draw()


# --- Main Application Run ---
if __name__ == "__main__":
    # IMPORTANT: Before running this script, ensure you have the following Python libraries installed:
    # pip install yfinance pandas numpy scikit-learn tensorflow vaderSentiment textblob matplotlib

    # For TextBlob, you might need to download NLTK corpora:
    # python -m textblob.download_corpora

    # Also, ensure Ollama is installed and running on your system,
    # and you have pulled the 'deepseek-r1' model (e.g., by running 'ollama pull deepseek-r1' in your terminal).

    root = tk.Tk()
    app = ForwardLookingStockPredictor(root)
    root.mainloop()