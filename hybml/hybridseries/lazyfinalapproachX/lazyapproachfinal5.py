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

# Ollama library - ensure Ollama server is running and 'llama3' model is pulled
import ollama


class ForwardLookingStockPredictor:
    """
    A Tkinter application for predicting stock prices using historical data,
    and influencing that prediction with recent news sentiment (VADER, TextBlob, or Llama3).
    The application features a tabbed UI for prediction results and detailed news sentiment.
    It predicts two future trading days, skipping the immediate next trading day.
    Includes enhanced Llama3 integration for news relevance filtering and dynamic prediction influence.
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

        tk.Label(self.input_frame, text="Company Ticker (e.g., AAPL):").grid(row=0, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.ticker_entry = tk.Entry(self.input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self.input_frame, text="Historical Data Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5,
                                                                                         pady=5, sticky="w")
        self.start_date_entry = tk.Entry(self.input_frame)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365 * 2)).strftime(
            '%Y-%m-%d'))  # Default 2 years back

        tk.Label(self.input_frame, text="Historical Data End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=5,
                                                                                       sticky="w")
        self.end_date_entry = tk.Entry(self.input_frame)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, (datetime.now() - timedelta(days=1)).strftime(
            '%Y-%m-%d'))  # End date is typically yesterday

        tk.Label(self.input_frame, text="LSTM Lookback Window (days):").grid(row=3, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.lookback_entry = tk.Entry(self.input_frame)
        self.lookback_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.lookback_entry.insert(0, "60")  # Default lookback window for LSTM

        tk.Label(self.input_frame, text="Neutral Trend Lookback Days:").grid(row=4, column=0, padx=5, pady=5,
                                                                             sticky="w")
        self.neutral_trend_lookback_entry = tk.Entry(self.input_frame)
        self.neutral_trend_lookback_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.neutral_trend_lookback_entry.insert(0, "5")  # Default lookback for neutral trend

        tk.Label(self.input_frame, text="Choose Sentiment Analyzer:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.analyzer_choice = ttk.Combobox(self.input_frame,
                                            values=["Llama3", "VADER", "TextBlob"],
                                            state="readonly")
        self.analyzer_choice.set("Llama3")  # Default choice
        self.analyzer_choice.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        self.predict_button = tk.Button(self.input_frame, text="Predict Prices & Plot",
                                        command=self._start_prediction_thread)
        self.predict_button.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")

        self.ollama_status_label = tk.Label(self.input_frame, text="Ollama Status: Checking...", fg="gray")
        self.ollama_status_label.grid(row=7, column=0, columnspan=2, pady=5, sticky="ew")

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

        if not self.cached_news_sentiment_details:
            self.news_details_text.insert(tk.END, "No recent news found or analyzed for the last prediction run.\n")
        else:
            self.news_details_text.insert(tk.END,
                                          f"Overall Average Sentiment Score (from relevant news): {self.cached_overall_sentiment:.4f}\n\n")
            for item in self.cached_news_sentiment_details:
                self.news_details_text.insert(tk.END, f"Title: {item['title']}\n")
                self.news_details_text.insert(tk.END, f"Summary: {item['summary']}\n")
                self.news_details_text.insert(tk.END,
                                              f"Individual Sentiment ({item['analyzer']}): {item['sentiment']:.4f}\n")
                self.news_details_text.insert(tk.END, "---\n\n")

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
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'hello'}], stream=True,
                                   options={'num_predict': 1})
            for chunk in response:
                pass
            self.master.after(0, lambda: self.ollama_status_label.config(text="Ollama Status: Running (llama3)",
                                                                         fg="green"))
        except Exception as e:
            self.master.after(0, lambda: self.ollama_status_label.config(
                text=f"Ollama Status: Not Running or Error ({e}). Ensure 'llama3' model is pulled.", fg="red"))

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
            messagebox.showwarning("Input Error", "Please fill in all fields.")
            return

        try:
            lookback_window = int(lookback_window_str)
            neutral_trend_lookback_days = int(neutral_trend_lookback_str)
            datetime.strptime(start_date_str, '%Y-%m-%d')
            datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            messagebox.showwarning("Input Error",
                                   "Lookback window and neutral trend days must be integers. Dates must be APAC-MM-DD.")
            return

        self.predict_button.config(state=tk.DISABLED)
        self._update_status("Starting prediction process...", "blue")
        self._update_log_display("", clear_previous=True)
        self.prediction_label.config(text="Prediction: Processing...")
        self.ax.clear()
        self.canvas.draw_idle()
        self.cached_news_sentiment_details = []  # Clear cached news
        self.cached_overall_sentiment = 0.0
        self._update_news_details_display()  # Clear news tab display

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
            stock_data = yf.download(ticker, start=start_date_str, end=end_date_dt, auto_adjust=False)
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

            # --- Step 2: Retrieve Recent News and Perform Sentiment Analysis (with ENHANCED Llama3 relevance filtering) ---
            self._update_status(
                f"Retrieving recent news for {ticker} and analyzing sentiment with {selected_analyzer}...", "blue")
            ticker_yf_obj = yf.Ticker(ticker)
            news_items = ticker_yf_obj.news

            company_info = ticker_yf_obj.info
            company_long_name = company_info.get('longName', ticker)

            # Llama3-based relevance filtering
            relevant_news_items = []
            total_news_checked = len(news_items)
            self._update_log_display(
                f"Found {total_news_checked} total recent news items. Filtering with Llama3 for relevance...",
                clear_previous=False)

            for i, item in enumerate(news_items):
                title = item.get('title') or item.get('headline') or 'No Title Provided'
                summary = item.get('content', {}).get('summary') or item.get('content', {}).get(
                    'description') or 'No Summary Provided'
                full_text_for_llama = f"Title: {title}\nSummary: {summary}".strip()

                if not full_text_for_llama:
                    continue  # Skip empty news items

                relevance_prompt = (
                    f"Is this news article directly and primarily about {company_long_name} ({ticker})? "
                    "Respond with 'YES' or 'NO'.\n\n"
                    f"News: {full_text_for_llama}"
                )
                try:
                    relevance_response = ollama.chat(
                        model='llama3', messages=[{'role': 'user', 'content': relevance_prompt}], stream=False,
                        options={'num_predict': 10}
                    )
                    relevance_answer = relevance_response['message']['content'].strip().upper()
                    if "YES" in relevance_answer:
                        relevant_news_items.append(item)
                except Exception as e:
                    self._update_log_display(f"Warning: Llama3 relevance check failed for news item {i + 1}: {e}",
                                             clear_previous=False)
                    # Fallback: if Llama3 fails, use keyword check as a last resort
                    # This fallback should still be strong, check both ticker and common name
                    fallback_keywords = [ticker.lower()]
                    if company_long_name and company_long_name != ticker:
                        fallback_keywords.append(company_long_name.lower().split(' ')[0])  # First word of long name
                        fallback_keywords.append(company_long_name.lower())  # Full long name

                    is_relevant_by_fallback = False
                    lower_full_text = full_text_for_llama.lower()
                    for keyword in set(fallback_keywords):  # Use set to avoid duplicates
                        if keyword in lower_full_text:
                            is_relevant_by_fallback = True
                            break
                    if is_relevant_by_fallback:
                        relevant_news_items.append(item)
                        self._update_log_display(f"Fallback: News item {i + 1} included due to keyword match.",
                                                 clear_previous=False)

            overall_latest_sentiment_score = 0.0
            individual_news_sentiment_data = []
            valid_sentiment_count = 0

            self._update_log_display(
                f"After Llama3 relevance filtering, found {len(relevant_news_items)} relevant news items.",
                clear_previous=False)

            for i, item in enumerate(relevant_news_items):
                title = item.get('title') or item.get('headline') or 'No Title Provided'
                summary = item.get('content', {}).get('summary') or item.get('content', {}).get(
                    'description') or 'No Summary Provided'
                full_text = f"{title}. {summary}".strip()

                current_item_sentiment = 0.0
                if selected_analyzer == "Llama3":
                    self._update_status(f"Llama3: Analyzing relevant news item {i + 1}/{len(relevant_news_items)}...",
                                        "blue")
                    ollama_prompt = (
                        f"Analyze this news headline and summary for sentiment towards the company ({ticker}). "
                        "Provide a numerical sentiment score between -1.0 (very negative) and +1.0 (very positive). "
                        "Format your response strictly as: Score: [SCORE]\n\n"
                        f"News: {full_text}"
                    )
                    try:
                        ollama_response = ollama.chat(
                            model='llama3', messages=[{'role': 'user', 'content': ollama_prompt}], stream=False
                        )
                        llama_output = ollama_response['message']['content'].strip()
                        score_match = re.search(r"Score: ([-+]?\d*\.?\d+)", llama_output)
                        if score_match:
                            current_item_sentiment = float(score_match.group(1))
                        else:
                            current_item_sentiment = 0.0
                            if "positive" in llama_output.lower():
                                current_item_sentiment = 0.7
                            elif "negative" in llama_output.lower():
                                current_item_sentiment = -0.7

                    except Exception as ollama_e:
                        current_item_sentiment = 0.0
                        self._update_log_display(
                            f"Warning: Llama3 sentiment analysis failed for item {i + 1}: {ollama_e}",
                            clear_previous=False)

                elif selected_analyzer == "VADER":
                    sentiment_scores = self.vader_analyzer.polarity_scores(full_text)
                    current_item_sentiment = sentiment_scores['compound']

                elif selected_analyzer == "TextBlob":
                    analysis = TextBlob(full_text)
                    current_item_sentiment = analysis.sentiment.polarity

                individual_news_sentiment_data.append({
                    'title': title,
                    'summary': summary,
                    'sentiment': current_item_sentiment,
                    'analyzer': selected_analyzer
                })
                overall_latest_sentiment_score += current_item_sentiment
                valid_sentiment_count += 1

            if valid_sentiment_count > 0:
                overall_latest_sentiment_score /= valid_sentiment_count
            else:
                overall_latest_sentiment_score = 0.0

            self.cached_news_sentiment_details = individual_news_sentiment_data
            self.cached_overall_sentiment = overall_latest_sentiment_score

            self._update_log_display(
                f"Overall Latest Sentiment Score (from relevant news): {overall_latest_sentiment_score:.4f}\n",
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

            # --- Step 6: Apply Sentiment-Based Adjustment or Base Trend to Each Prediction (ENHANCED Llama3 Influence) ---
            last_actual_price = stock_price_df['Close'].iloc[-1]

            # Calculate predicted dates: skip immediate next trading day, then predict 2 days
            predicted_dates_ts = []
            current_date_ts = stock_price_df.index.max()  # Start from last historical date

            # Skip the immediate next trading day: Advance one day and skip weekends
            current_date_ts += timedelta(days=1)  # Move to the literal next calendar day
            while current_date_ts.weekday() >= 5:  # Skip weekends (Sat=5, Sun=6)
                current_date_ts += timedelta(days=1)
            # Now current_date_ts is the *first trading day to skip*

            # Now, calculate the NEXT 2 trading days after this skipped day
            for _ in range(2):
                current_date_ts += timedelta(days=1)
                while current_date_ts.weekday() >= 5:  # Skip weekends (Sat=5, Sun=6)
                    current_date_ts += timedelta(days=1)
                predicted_dates_ts.append(current_date_ts)

            predicted_dates_for_display = [dt.to_pydatetime() for dt in predicted_dates_ts]

            final_predicted_prices = []
            influence_explanation = ""  # Detailed explanation of Llama3's influence

            # Llama3 to classify overall market impact from the sentiment and news
            news_titles_for_llama = []
            if individual_news_sentiment_data:
                for item in individual_news_sentiment_data:
                    news_titles_for_llama.append(f" - {item['title']}")

            # Join titles with actual newline characters
            news_titles_str = '\n'.join(news_titles_for_llama)

            llama_impact_prompt = (
                f"Given the following combined news sentiment score for {ticker} is {overall_latest_sentiment_score:.4f} and considering the latest relevant news headlines:\n\n"
                f"{news_titles_str}\n\n"  # Correctly using pre-joined string here
                "How would you classify the overall likely impact of this news on its stock price? "
                "Choose one of the following precise classifications: 'Strong Positive', 'Moderate Positive', 'Neutral', 'Moderate Negative', 'Strong Negative'."
            )

            # Default to Neutral if no relevant news or Llama3 classification fails
            overall_impact_classification = "Neutral"
            try:
                if individual_news_sentiment_data:  # Only call Llama3 for impact if there's relevant news
                    self._update_status("Llama3: Classifying overall news impact...", "blue")
                    impact_response = ollama.chat(
                        model='llama3', messages=[{'role': 'user', 'content': llama_impact_prompt}], stream=False,
                        options={'num_predict': 20}
                    )
                    impact_text = impact_response['message']['content'].strip().upper()
                    if "STRONG POSITIVE" in impact_text:
                        overall_impact_classification = "Strong Positive"
                    elif "MODERATE POSITIVE" in impact_text:
                        overall_impact_classification = "Moderate Positive"
                    elif "STRONG NEGATIVE" in impact_text:
                        overall_impact_classification = "Strong Negative"
                    elif "MODERATE NEGATIVE" in impact_text:
                        overall_impact_classification = "Moderate Negative"
                    else:  # Default to neutral if no clear classification, or "Neutral" is in the text
                        overall_impact_classification = "Neutral"
                else:
                    influence_explanation = "No relevant news found, defaulting to Neutral impact."
            except Exception as e:
                self._update_log_display(
                    f"Warning: Llama3 impact classification failed: {e}. Defaulting to Neutral impact.",
                    clear_previous=False)
                overall_impact_classification = "Neutral"

            # Define dynamic adjustment percentages based on Llama3's classification
            impact_adjustments = {
                "Strong Positive": 0.02,  # 2% upward adjustment
                "Moderate Positive": 0.008,  # 0.8% upward adjustment
                "Neutral": 0,  # Handled by base trend
                "Moderate Negative": -0.008,  # 0.8% downward adjustment
                "Strong Negative": -0.02  # 2% downward adjustment
            }

            sentiment_adjustment_percentage = impact_adjustments.get(overall_impact_classification, 0)

            for raw_predicted_price in raw_predicted_actual_prices:
                adjusted_price = raw_predicted_price

                if overall_impact_classification == "Neutral":
                    if len(stock_price_df) < neutral_trend_lookback_days:
                        influence_explanation = "Neutral impact classified. Not enough data for base trend. Prediction relies on LSTM trend."
                    else:
                        recent_prices = stock_price_df['Close'].tail(neutral_trend_lookback_days)
                        if len(recent_prices) > 1:
                            daily_returns = recent_prices.pct_change().dropna()
                            if not daily_returns.empty:
                                avg_daily_return = daily_returns.mean()
                                base_trend_adjustment_amount = raw_predicted_price * avg_daily_return
                                adjusted_price = raw_predicted_price + base_trend_adjustment_amount
                                influence_explanation = (
                                    f"Neutral impact classified. Adjusting predictions by base trend ({avg_daily_return:.2%}) from last {neutral_trend_lookback_days} days."
                                )
                            else:
                                influence_explanation = f"Neutral impact classified. No valid daily returns for base trend calculation. Prediction relies on LSTM trend."
                        else:
                            influence_explanation = f"Neutral impact classified. Not enough data points to calculate base trend. Prediction relies on LSTM trend."
                else:
                    adjustment_amount = raw_predicted_price * sentiment_adjustment_percentage
                    adjusted_price += adjustment_amount
                    influence_explanation = (
                        f"Llama3 classified '{overall_impact_classification}' impact. Adjusting predictions by "
                        f"{abs(sentiment_adjustment_percentage * 100):.1f}%."
                    )

                final_predicted_prices.append(max(0.01, adjusted_price))

            # --- Final Display ---
            prediction_text = (
                f"Prediction for {ticker}:\n"
                f"Last Known Actual Close Price: ${last_actual_price:.2f}\n"
                f"Raw LSTM Predicted Day 1 ({predicted_dates_for_display[0].strftime('%Y-%m-%d')}): ${raw_predicted_actual_prices[0]:.2f}\n"
                f"Raw LSTM Predicted Day 2 ({predicted_dates_for_display[1].strftime('%Y-%m-%d')}): ${raw_predicted_actual_prices[1]:.2f}\n"
                f"Overall News Sentiment (via {selected_analyzer}): {overall_latest_sentiment_score:.4f}\n"
                f"Llama3 Classified Impact: {overall_impact_classification}\n"
                f"Influence on Prediction: {influence_explanation}\n"
                f"Final Adjusted Predicted Day 1 Price: ${final_predicted_prices[0]:.2f}\n"
                f"Final Adjusted Predicted Day 2 Price: ${final_predicted_prices[1]:.2f}"
            )
            self.master.after(0, lambda: self.prediction_label.config(text=prediction_text, fg="darkgreen"))
            self.master.after(0, lambda: messagebox.showinfo("Prediction Complete",
                                                             "Stock prices for next 2 trading days (skipping tomorrow) predicted with news sentiment influence!"))
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
            self.master.after(0, lambda e_val=caught_e: self.prediction_label.config(text=f"Prediction Failed: {e_val}",
                                                                                     fg="red"))
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
    # and you have pulled the 'llama3' model (e.g., by running 'ollama pull llama3' in your terminal).

    root = tk.Tk()
    app = ForwardLookingStockPredictor(root)
    root.mainloop()

