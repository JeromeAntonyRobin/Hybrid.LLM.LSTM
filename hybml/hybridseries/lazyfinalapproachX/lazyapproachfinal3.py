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
    A Tkinter application for predicting the next day's stock price using historical data,
    and influencing that prediction with recent news sentiment (VADER, TextBlob, or Llama3).
    The application plots historical prices and extends the graph with the sentiment-adjusted prediction.
    Now includes a base trend for neutral sentiment.
    """
    def __init__(self, master):
        self.master = master
        master.title("Forward-Looking Stock Predictor with Sentiment")
        master.geometry("1100x850") # Larger window for plot and info
        master.resizable(True, True)

        # Main frame layout
        main_frame = tk.Frame(master)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1) # Left panel (inputs, logs)
        main_frame.grid_columnconfigure(1, weight=1) # Right panel (plot)
        main_frame.grid_rowconfigure(0, weight=0)    # Input frame at top
        main_frame.grid_rowconfigure(1, weight=1)    # Logs and plot share space

        # --- Input Parameters Frame ---
        self.input_frame = tk.LabelFrame(main_frame, text="Prediction Parameters", padx=10, pady=10)
        self.input_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        tk.Label(self.input_frame, text="Company Ticker (e.g., AAPL):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = tk.Entry(self.input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self.input_frame, text="Historical Data Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = tk.Entry(self.input_frame)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365 * 2)).strftime('%Y-%m-%d')) # Default 2 years back

        tk.Label(self.input_frame, text="Historical Data End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = tk.Entry(self.input_frame)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')) # End date is typically yesterday

        tk.Label(self.input_frame, text="LSTM Lookback Window (days):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.lookback_entry = tk.Entry(self.input_frame)
        self.lookback_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.lookback_entry.insert(0, "60") # Default lookback window for LSTM

        tk.Label(self.input_frame, text="Neutral Trend Lookback Days:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.neutral_trend_lookback_entry = tk.Entry(self.input_frame)
        self.neutral_trend_lookback_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.neutral_trend_lookback_entry.insert(0, "5") # Default lookback for neutral trend

        tk.Label(self.input_frame, text="Choose Sentiment Analyzer:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.analyzer_choice = ttk.Combobox(self.input_frame, 
                                             values=["Llama3", "VADER", "TextBlob"], 
                                             state="readonly")
        self.analyzer_choice.set("Llama3") # Default choice
        self.analyzer_choice.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
        
        self.predict_button = tk.Button(self.input_frame, text="Predict Next Day Price & Plot", command=self._start_prediction_thread)
        self.predict_button.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")

        self.ollama_status_label = tk.Label(self.input_frame, text="Ollama Status: Checking...", fg="gray")
        self.ollama_status_label.grid(row=7, column=0, columnspan=2, pady=5, sticky="ew")
        
        # --- Left Panel: Logs and Prediction Results ---
        self.left_panel_frame = tk.Frame(main_frame)
        self.left_panel_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.left_panel_frame.grid_rowconfigure(0, weight=0) # Prediction label fixed
        self.left_panel_frame.grid_rowconfigure(1, weight=1) # Logs expand
        self.left_panel_frame.grid_columnconfigure(0, weight=1)

        self.prediction_label = tk.Label(self.left_panel_frame, text="Prediction: Awaiting inputs...", wraplength=480, justify=tk.LEFT, font=("Arial", 11, "bold"))
        self.prediction_label.grid(row=0, column=0, padx=5, pady=5, sticky="nw")

        self.log_frame = tk.LabelFrame(self.left_panel_frame, text="Processing Logs", padx=5, pady=5)
        self.log_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.log_frame.grid_rowconfigure(0, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)

        self.log_text_display = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, width=60, height=15, state='disabled')
        self.log_text_display.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # --- Right Panel: Plotting Frame ---
        self.plot_frame = tk.LabelFrame(main_frame, text="Historical Data & Predicted Price", padx=5, pady=5)
        self.plot_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.toolbar_frame = tk.Frame(self.plot_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # --- Status Bar ---
        self.status_label = tk.Label(master, text="Ready.", relief=tk.SUNKEN, bd=1, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Initialize analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initial Ollama status check in a separate thread
        self._start_ollama_status_check()

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
        self.log_text_display.see(tk.END) # Scroll to the end
        self.log_text_display.config(state='disabled')

    def _start_ollama_status_check(self):
        """Starts a thread to check Ollama status."""
        thread = threading.Thread(target=self._check_ollama_status_task)
        thread.daemon = True
        thread.start()

    def _check_ollama_status_task(self):
        """Task to check Ollama server status and update GUI."""
        try:
            # Send a minimal prompt to check connectivity and 'llama3' model availability
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'hello'}], stream=True, options={'num_predict': 1})
            for chunk in response: # Iterate to ensure response is consumed
                pass
            self.master.after(0, lambda: self.ollama_status_label.config(text="Ollama Status: Running (llama3)", fg="green"))
        except Exception as e:
            self.master.after(0, lambda: self.ollama_status_label.config(text=f"Ollama Status: Not Running or Error ({e}). Ensure 'llama3' model is pulled.", fg="red"))

    def _start_prediction_thread(self):
        """Starts a new thread for the entire prediction process."""
        ticker = self.ticker_entry.get().strip().upper()
        start_date_str = self.start_date_entry.get()
        end_date_str = self.end_date_entry.get()
        lookback_window_str = self.lookback_entry.get()
        neutral_trend_lookback_str = self.neutral_trend_lookback_entry.get() # New input
        selected_analyzer = self.analyzer_choice.get()

        if not all([ticker, start_date_str, end_date_str, lookback_window_str, neutral_trend_lookback_str, selected_analyzer]):
            messagebox.showwarning("Input Error", "Please fill in all fields.")
            return

        try:
            lookback_window = int(lookback_window_str)
            neutral_trend_lookback_days = int(neutral_trend_lookback_str)
            datetime.strptime(start_date_str, '%Y-%m-%d')
            datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            messagebox.showwarning("Input Error", "Lookback window and neutral trend days must be integers. Dates must be YYYY-MM-DD.")
            return

        self.predict_button.config(state=tk.DISABLED)
        self._update_status("Starting prediction process...", "blue")
        self._update_log_display("", clear_previous=True) # Clear previous logs
        self.prediction_label.config(text="Prediction: Processing...")
        self.ax.clear() # Clear any previous plot
        self.canvas.draw_idle()

        thread = threading.Thread(target=self._perform_prediction_task, 
                                  args=(ticker, start_date_str, end_date_str, 
                                        lookback_window, neutral_trend_lookback_days, 
                                        selected_analyzer))
        thread.daemon = True
        thread.start()

    def _perform_prediction_task(self, ticker, start_date_str, end_date_str, lookback_window, neutral_trend_lookback_days, selected_analyzer):
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
            self._update_status(f"Downloading historical stock data for {ticker} from {start_date_str} to {end_date_str}...", "blue")
            # Set auto_adjust=False for robustness, as True sometimes creates MultiIndex columns with ticker.
            stock_data = yf.download(ticker, start=start_date_str, end=end_date_str, auto_adjust=False)
            if stock_data.empty:
                raise ValueError(f"No stock data found for {ticker} in the specified date range. Please check ticker or dates.")

            # --- CRITICAL FIX: Flatten MultiIndex Columns to single strings ---
            flattened_columns = []
            for col in stock_data.columns:
                if isinstance(col, tuple):
                    flattened_columns.append(col[0])
                else:
                    flattened_columns.append(col)
            stock_data.columns = flattened_columns
            # --- END CRITICAL FIX ---

            # Ensure 'Close' or 'Adj Close' exists after flattening and rename to 'Close'
            stock_price_df = None
            if 'Close' in stock_data.columns:
                stock_price_df = stock_data[['Close']].copy()
            elif 'Adj Close' in stock_data.columns:
                stock_price_df = stock_data[['Adj Close']].copy()
                stock_price_df.columns = ['Close'] # Standardize to 'Close'
            else:
                raise KeyError(f"Could not find 'Close' or 'Adj Close' column after processing for {ticker}. Available columns: {stock_data.columns.tolist()}")
            
            # Ensure index is DatetimeIndex and normalized
            stock_price_df.index = pd.to_datetime(stock_price_df.index).normalize()
            stock_price_df.index.name = 'date'
            stock_price_df.sort_index(inplace=True)

            self._update_log_display(f"Stock data head:\n{stock_price_df.head().to_string()}\n")
            self._update_log_display(f"Stock data tail:\n{stock_price_df.tail().to_string()}\n", clear_previous=False)
            self._update_log_display(f"Stock data index type: {type(stock_price_df.index)}\n", clear_previous=False)
            self._update_log_display(f"Stock data columns: {stock_price_df.columns.tolist()}\n", clear_previous=False)


            # --- Step 2: Retrieve Recent News and Perform Sentiment Analysis ---
            self._update_status(f"Retrieving recent news for {ticker} and analyzing sentiment with {selected_analyzer}...", "blue")
            ticker_yf_obj = yf.Ticker(ticker)
            news_items = ticker_yf_obj.news

            latest_sentiment_score = 0.0
            latest_news_headlines_list = []
            valid_sentiment_count = 0

            if not news_items:
                self._update_log_display("No recent news found for the ticker. Latest sentiment will be neutral.\n", clear_previous=False)
            else:
                self._update_log_display(f"Found {len(news_items)} recent news items.", clear_previous=False)

                for i, item in enumerate(news_items):
                    content_data = item.get('content', {})
                    title = content_data.get('title') or content_data.get('headline') or 'No Title Provided'
                    summary = content_data.get('summary') or content_data.get('description') or 'No Summary Provided'
                    
                    if title == 'No Title Provided' and summary == 'No Summary Provided':
                        continue

                    full_text = f"{title}. {summary}".strip()
                    latest_news_headlines_list.append(f"Title: {title}\nSummary: {summary}\n---") # Collect for display
                    
                    current_item_sentiment = 0.0
                    if selected_analyzer == "Llama3":
                        self._update_status(f"Llama3: Analyzing news item {i+1}/{len(news_items)}...", "blue")
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
                                current_item_sentiment = 0.0 # Default to neutral if parsing fails
                                if "positive" in llama_output.lower(): current_item_sentiment = 0.7
                                elif "negative" in llama_output.lower(): current_item_sentiment = -0.7
                                
                        except Exception as ollama_e:
                            current_item_sentiment = 0.0 # Default to neutral on error
                            self._update_log_display(f"Warning: Llama3 analysis failed for item {i+1}: {ollama_e}", clear_previous=False)

                    elif selected_analyzer == "VADER":
                        sentiment_scores = self.vader_analyzer.polarity_scores(full_text)
                        current_item_sentiment = sentiment_scores['compound']

                    elif selected_analyzer == "TextBlob":
                        analysis = TextBlob(full_text)
                        current_item_sentiment = analysis.sentiment.polarity
                    
                    latest_sentiment_score += current_item_sentiment
                    valid_sentiment_count += 1
            
            if valid_sentiment_count > 0:
                latest_sentiment_score /= valid_sentiment_count # Calculate average of recent news
            else:
                latest_sentiment_score = 0.0 # Default to neutral if no news found or analyzed

            latest_news_display = "\n".join(latest_news_headlines_list)
            self._update_log_display(f"\n--- Latest News for Sentiment Analysis ({selected_analyzer}) ---\n"
                                     f"{latest_news_display}\n"
                                     f"Overall Latest Sentiment Score: {latest_sentiment_score:.4f}\n", clear_previous=False)

            # --- Step 3: Prepare Data for LSTM (Historical Prices Only) ---
            self._update_status("Preparing historical data for LSTM training...", "blue")
            
            # LSTM will train only on historical 'Close' prices
            data_for_lstm = stock_price_df['Close'].values.reshape(-1, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_for_lstm)

            X, y = [], []
            if len(scaled_data) < lookback_window + 1:
                raise ValueError(f"Not enough historical data ({len(scaled_data)} data points) for lookback window of {lookback_window} days. Please extend date range.")

            for i in range(lookback_window, len(scaled_data)):
                X.append(scaled_data[i-lookback_window:i, 0]) # Sequence of past 'Close' prices
                y.append(scaled_data[i, 0]) # Next 'Close' price

            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # Reshape for LSTM: [samples, time_steps, features=1]

            # --- Step 4: Train LSTM Model ---
            self._update_status("Training LSTM model...", "blue")
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1))) # input_shape: (time_steps, features)
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stopping])
            self._update_status("LSTM model trained successfully!", "green")

            # --- Step 5: Predict Raw Next Day Price ---
            self._update_status("Predicting next day's raw stock price...", "blue")
            
            # Get the last 'lookback_window' days of historical prices for prediction input
            last_lookback_prices = scaled_data[-lookback_window:].reshape(1, lookback_window, 1)
            raw_predicted_scaled_price = model.predict(last_lookback_prices, verbose=0)
            
            # Inverse transform the prediction to actual price scale
            raw_predicted_price = scaler.inverse_transform(raw_predicted_scaled_price)[0, 0]

            # --- Step 6: Apply Sentiment-Based Adjustment or Base Trend ---
            last_actual_price = stock_price_df['Close'].iloc[-1]
            prediction_date_ts = stock_price_df.index.max() + timedelta(days=1)
            
            # Find the next *trading* day
            while prediction_date_ts.weekday() >= 5: # Monday=0, Sunday=6
                prediction_date_ts += timedelta(days=1)

            prediction_date_for_display = prediction_date_ts.to_pydatetime()


            final_predicted_price = raw_predicted_price
            sentiment_influence_msg = ""
            
            sentiment_positive_threshold = 0.2
            sentiment_negative_threshold = -0.2
            sentiment_neutral_min_threshold = -0.1 # Define neutral range
            sentiment_neutral_max_threshold = 0.1

            # Adjustment for non-neutral sentiment
            sentiment_adjustment_percentage = 0.008 # 0.8% adjustment based on raw predicted price

            if sentiment_neutral_min_threshold <= latest_sentiment_score <= sentiment_neutral_max_threshold:
                # --- Neutral Sentiment: Follow Base Trend ---
                if len(stock_price_df) < neutral_trend_lookback_days:
                    # Not enough data for base trend, revert to no adjustment
                    sentiment_influence_msg = f"Neutral News ({selected_analyzer} score: {latest_sentiment_score:.2f}). Not enough data for base trend. Prediction relies on LSTM trend."
                else:
                    # Calculate average daily percentage change over neutral_trend_lookback_days
                    recent_prices = stock_price_df['Close'].tail(neutral_trend_lookback_days)
                    # Check if there are at least two points to calculate percentage change
                    if len(recent_prices) > 1:
                        daily_returns = recent_prices.pct_change().dropna()
                        if not daily_returns.empty:
                            avg_daily_return = daily_returns.mean()
                            base_trend_adjustment_amount = raw_predicted_price * avg_daily_return
                            final_predicted_price = raw_predicted_price + base_trend_adjustment_amount
                            sentiment_influence_msg = (
                                f"Neutral News ({selected_analyzer} score: {latest_sentiment_score:.2f}). "
                                f"Adjusting prediction by base trend ({avg_daily_return:.2%}) from last {neutral_trend_lookback_days} days."
                            )
                        else:
                            sentiment_influence_msg = f"Neutral News ({selected_analyzer} score: {latest_sentiment_score:.2f}). No valid daily returns for base trend calculation. Prediction relies on LSTM trend."
                    else:
                        sentiment_influence_msg = f"Neutral News ({selected_analyzer} score: {latest_sentiment_score:.2f}). Not enough data points to calculate base trend. Prediction relies on LSTM trend."
            elif latest_sentiment_score > sentiment_positive_threshold:
                # --- Positive Sentiment: Adjust Upwards ---
                sentiment_influence_msg = f"Strong Positive News ({selected_analyzer} score: {latest_sentiment_score:.2f}). Adjusting prediction upwards by {sentiment_adjustment_percentage*100:.1f}%."
                final_predicted_price += abs(raw_predicted_price * sentiment_adjustment_percentage)
            elif latest_sentiment_score < sentiment_negative_threshold:
                # --- Negative Sentiment: Adjust Downwards ---
                sentiment_influence_msg = f"Strong Negative News ({selected_analyzer} score: {latest_sentiment_score:.2f}). Adjusting prediction downwards by {sentiment_adjustment_percentage*100:.1f}%."
                final_predicted_price -= abs(raw_predicted_price * sentiment_adjustment_percentage)
            else:
                # --- Mildly Neutral/Slightly Positive/Negative (outside defined thresholds but not strongly positive/negative) ---
                sentiment_influence_msg = f"Mild News ({selected_analyzer} score: {latest_sentiment_score:.2f}). Prediction relies primarily on LSTM trend."
                # No additional adjustment, raw_predicted_price is used as final_predicted_price

            # Ensure price doesn't go below zero
            final_predicted_price = max(0.01, final_predicted_price)


            # --- Final Display ---
            prediction_text = (
                f"Prediction for {prediction_date_for_display.strftime('%Y-%m-%d')} for {ticker}:\n"
                f"Last Known Actual Close Price: ${last_actual_price:.2f}\n"
                f"Raw LSTM Predicted Close Price: ${raw_predicted_price:.2f}\n"
                f"Latest Combined News Sentiment ({selected_analyzer}): {latest_sentiment_score:.4f}\n"
                f"Influence on Prediction: {sentiment_influence_msg}\n" # Updated message
                f"Final Adjusted Predicted Close Price: ${final_predicted_price:.2f}"
            )
            self.master.after(0, lambda: self.prediction_label.config(text=prediction_text, fg="darkgreen"))
            self.master.after(0, lambda: messagebox.showinfo("Prediction Complete", "Next day stock price predicted with news sentiment influence!"))
            self._update_status("Prediction process complete!", "green")

            # --- Step 7: Plotting Historical Data and Future Prediction ---
            self._update_status("Generating plot of historical data and future prediction...", "blue")
            self._plot_prediction_and_history(ticker, stock_price_df, prediction_date_for_display, final_predicted_price)
            self._update_status("Plot generated! Check the right panel.", "green")

        except Exception as caught_e:
            error_traceback = traceback.format_exc()
            error_message = f"An error occurred: {type(caught_e).__name__}: {caught_e}\n\nTraceback:\n{error_traceback}"
            
            self.master.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {caught_e}"))
            self.master.after(0, lambda msg=error_message: self._update_log_display(msg, clear_previous=False)) # Add to logs
            self.master.after(0, lambda: self._update_status("Error during prediction. Check logs.", "red"))
            self.master.after(0, lambda e_val=caught_e: self.prediction_label.config(text=f"Prediction Failed: {e_val}", fg="red"))
        finally:
            self.master.after(0, lambda: self.predict_button.config(state=tk.NORMAL))

    def _plot_prediction_and_history(self, ticker, historical_df, predicted_date, predicted_price):
        """
        Plots the historical stock prices and extends the graph with the single
        sentiment-adjusted future predicted price.
        """
        self.ax.clear()
        
        # Plot historical data
        self.ax.plot(historical_df.index, historical_df['Close'], label='Historical Close Price', color='blue')
        
        # Add the predicted point as an extension
        last_historical_date = historical_df.index.max()
        
        # Create a line segment from the last historical point to the predicted point
        # Ensure both dates are standard datetime objects for matplotlib to handle correctly
        plot_dates = [last_historical_date.to_pydatetime(), predicted_date] 
        plot_prices = [historical_df['Close'].iloc[-1], predicted_price]
        self.ax.plot(plot_dates, plot_prices, 'ro--', label='Predicted Next Day Price', markersize=8) # Red circle for prediction

        self.ax.set_title(f'{ticker} Stock Price: Historical & Predicted (Sentiment-Adjusted)')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.autofmt_xdate() # Rotate dates for better readability
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

