import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Ollama library (ensure Ollama server is running and a model is pulled)
import ollama


class StockNewsPredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Stock News Predictor with Sentiment & LSTM")

        # --- Input Frame ---
        self.input_frame = tk.LabelFrame(master, text="Input Parameters")
        self.input_frame.pack(padx=10, pady=10, fill="x")

        self.ticker_label = tk.Label(self.input_frame, text="Company Ticker (e.g., AAPL):")
        self.ticker_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = tk.Entry(self.input_frame)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.start_date_label = tk.Label(self.input_frame, text="Start Date (YYYY-MM-DD):")
        self.start_date_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = tk.Entry(self.input_frame)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))

        self.end_date_label = tk.Label(self.input_frame, text="End Date (YYYY-MM-DD):")
        self.end_date_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = tk.Entry(self.input_frame)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))

        self.lookback_label = tk.Label(self.input_frame, text="LSTM Lookback Window:")
        self.lookback_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.lookback_entry = tk.Entry(self.input_frame)
        self.lookback_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.lookback_entry.insert(0, "60")

        self.get_data_button = tk.Button(self.input_frame, text="Get Data & Process", command=self.get_data_and_process)
        self.get_data_button.grid(row=4, column=0, columnspan=2, pady=10)

        # --- Data Display Frame ---
        self.data_frame = tk.LabelFrame(master, text="Processed Data & CSV Export")
        self.data_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.data_text = scrolledtext.ScrolledText(self.data_frame, wrap=tk.WORD, width=80, height=15)
        self.data_text.pack(padx=5, pady=5, fill="both", expand=True)

        self.export_csv_button = tk.Button(self.data_frame, text="Export Combined CSV",
                                           command=self.export_combined_csv)
        self.export_csv_button.pack(pady=5)

        # --- LSTM & Prediction Frame ---
        self.prediction_frame = tk.LabelFrame(master, text="LSTM Prediction with Ollama Influence")
        self.prediction_frame.pack(padx=10, pady=10, fill="x")

        self.train_predict_button = tk.Button(self.prediction_frame, text="Train LSTM & Predict",
                                              command=self.train_and_predict)
        self.train_predict_button.pack(pady=5)

        self.prediction_label = tk.Label(self.prediction_frame, text="Prediction: Awaiting training...")
        self.prediction_label.pack(pady=5)

        self.ollama_status_label = tk.Label(self.prediction_frame, text="Ollama Status: Not Checked")
        self.ollama_status_label.pack(pady=5)
        self.check_ollama_status()

        self.data_combined_for_lstm = pd.DataFrame()

    def check_ollama_status(self):
        try:
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'hello'}], stream=True,
                                   options={'num_predict': 1})
            for chunk in response:
                pass
            self.ollama_status_label.config(text="Ollama Status: Running (llama3)")
            self.ollama_status_label.config(fg="green")
        except Exception as e:
            self.ollama_status_label.config(text=f"Ollama Status: Not Running or Error ({e})")
            self.ollama_status_label.config(fg="red")

    def get_data_and_process(self):
        ticker = self.ticker_entry.get().upper()
        start_date_str = self.start_date_entry.get()
        end_date_str = self.end_date_entry.get()

        if not ticker or not start_date_str or not end_date_str:
            messagebox.showerror("Input Error", "Please fill in all fields.")
            return

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            if start_date >= end_date:
                messagebox.showerror("Input Error", "Start date must be before end date.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Invalid date format. Please use YYYY-MM-DD.")
            return

        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, f"Retrieving data for {ticker} from {start_date_str} to {end_date_str}...\n")
        self.master.update_idletasks()

        try:
            # 1. Retrieve Stock Data (yfinance)
            self.data_text.insert(tk.END, f"Downloading stock data for {ticker}...\n")
            stock_data = yf.download(ticker, start=start_date_str, end=end_date_str)
            if stock_data.empty:
                self.data_text.insert(tk.END,
                                      "No stock data found for the given ticker and date range. Please check the ticker or date range.\n")
                messagebox.showerror("Data Error", "No stock data found. Please check ticker or dates.")
                return

            stock_data.index = pd.to_datetime(stock_data.index).normalize()
            stock_data_df = stock_data[['Close']].copy()

            self.data_text.insert(tk.END, "\n--- Stock Data Retrieved (sample) ---\n")
            self.data_text.insert(tk.END, stock_data_df.head().to_string() + "\n...\n")

            # 2. Retrieve Recent News Data (yfinance news) - ONLY YFINANCE IS USED FOR NEWS
            self.data_text.insert(tk.END, f"Retrieving recent news for {ticker} from yfinance...\n")
            self.master.update_idletasks()

            ticker_yf = yf.Ticker(ticker)
            news_items = ticker_yf.news

            parsed_news = []
            for item in news_items:
                # Safely access 'providerPublishTime' and convert
                if 'providerPublishTime' in item and item['providerPublishTime'] is not None:
                    try:
                        news_date = datetime.fromtimestamp(item['providerPublishTime']).normalize()
                        # Only consider news within the requested date range for consistency, though yfinance news is usually recent
                        if start_date <= news_date <= end_date:
                            combined_text = f"{item.get('title', '')}. {item.get('summary', '')}"
                            parsed_news.append({'date': news_date, 'text': combined_text})
                    except (ValueError, TypeError) as e:
                        self.data_text.insert(tk.END,
                                              f"Skipping news item due to invalid timestamp or format: {e} - {item.get('title', 'N/A')}\n")
                else:
                    self.data_text.insert(tk.END,
                                          f"Skipping news item: 'providerPublishTime' missing or null for '{item.get('title', 'N/A')}'\n")

            news_df = pd.DataFrame()
            if parsed_news:
                news_df = pd.DataFrame(parsed_news)
                news_df['date'] = pd.to_datetime(news_df['date'])

                news_df = news_df.groupby('date')['text'].apply(lambda x: ' '.join(x)).reset_index()
                news_df.set_index('date', inplace=True)
                news_df.rename(columns={'text': 'headlines'}, inplace=True)

            if news_df.empty:
                self.data_text.insert(tk.END,
                                      "No recent yfinance news found for the given ticker in the retrieved period within the specified date range.\n")
                news_df_for_merge = pd.DataFrame(index=stock_data_df.index)
                news_df_for_merge['headlines'] = "No news found for this day."
            else:
                news_df_for_merge = news_df.copy()

            self.data_text.insert(tk.END, "\n--- News Data Retrieved (sample) ---\n")
            if not news_df_for_merge.empty:
                self.data_text.insert(tk.END, news_df_for_merge.head().to_string() + "\n...\n")
            else:
                self.data_text.insert(tk.END, "No relevant yfinance news after processing.\n")

            # 3. Data Pairing
            stock_data_df.index = pd.to_datetime(stock_data_df.index)
            news_df_for_merge.index = pd.to_datetime(news_df_for_merge.index)

            self.data_combined_for_lstm = pd.merge(stock_data_df, news_df_for_merge, left_index=True, right_index=True,
                                                   how='left')
            self.data_combined_for_lstm.index.name = 'date'
            self.data_combined_for_lstm.rename(columns={'Close': 'stock_price'}, inplace=True)

            self.data_combined_for_lstm['headlines'] = self.data_combined_for_lstm['headlines'].fillna(
                "No news found for this day.")

            # 4. VADER Sentiment Analysis
            analyzer = SentimentIntensityAnalyzer()
            self.data_combined_for_lstm['sentiment_score'] = self.data_combined_for_lstm['headlines'].apply(
                lambda text: analyzer.polarity_scores(text)['compound']
            )

            self.data_combined_for_lstm = self.data_combined_for_lstm[['stock_price', 'sentiment_score', 'headlines']]

            self.data_text.insert(tk.END, "\n--- Combined Data with Sentiment (sample) ---\n")
            self.data_text.insert(tk.END, self.data_combined_for_lstm.head().to_string() + "\n...\n")

            messagebox.showinfo("Success", "Data retrieval, processing, and sentiment analysis complete!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during data processing: {e}")

    def export_combined_csv(self):
        if self.data_combined_for_lstm.empty:
            messagebox.showwarning("No Data", "Please retrieve and process data first.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                                                 initialfile="combined_stock_news.csv")
        if file_path:
            try:
                self.data_combined_for_lstm.to_csv(file_path)
                messagebox.showinfo("Export Success", f"Data successfully exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export CSV: {e}")

    def train_and_predict(self):
        if self.data_combined_for_lstm.empty:
            messagebox.showwarning("No Data", "Please retrieve and process data first before training.")
            return

        try:
            lookback_window = int(self.lookback_entry.get())
            if lookback_window <= 0:
                raise ValueError("Lookback window must be a positive integer.")
        except ValueError:
            messagebox.showerror("Input Error", "Invalid lookback window. Please enter a positive integer.")
            return

        self.prediction_label.config(text="Prediction: Training LSTM model...")
        self.master.update_idletasks()

        if len(self.data_combined_for_lstm) < lookback_window + 1:
            messagebox.showwarning("Training Error",
                                   "Not enough data points after merging for the specified lookback window. Try a smaller window or longer date range.")
            self.prediction_label.config(text="Prediction: Training failed (insufficient data).")
            return

        data = self.data_combined_for_lstm[['stock_price', 'sentiment_score']].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(lookback_window, len(scaled_data)):
            X.append(scaled_data[i - lookback_window:i, :])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            messagebox.showwarning("Training Error",
                                   "Not enough data to create sequences for LSTM training with the given lookback window.")
            self.prediction_label.config(text="Prediction: Training failed (insufficient data).")
            return

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                            callbacks=[early_stopping], verbose=0)

        self.prediction_label.config(text="Prediction: LSTM model trained. Calculating 'base trend' and predicting...")
        self.master.update_idletasks()

        train_predictions_scaled = model.predict(X_train)
        dummy_array_for_inverse_transform_train = np.zeros((len(train_predictions_scaled), scaled_data.shape[1]))
        dummy_array_for_inverse_transform_train[:, 0] = train_predictions_scaled.flatten()
        train_predictions = scaler.inverse_transform(dummy_array_for_inverse_transform_train)[:, 0]

        if len(train_predictions) > 1:
            base_trend_daily_change = np.mean(np.diff(train_predictions))
        else:
            base_trend_daily_change = 0

        # --- Prediction for the next day ---
        last_n_days_data = self.data_combined_for_lstm.tail(lookback_window)[['stock_price', 'sentiment_score']].values

        if len(last_n_days_data) < lookback_window:
            messagebox.showwarning("Prediction Error",
                                   "Not enough recent historical data to make a prediction for the next day with the chosen lookback window.")
            self.prediction_label.config(text="Prediction: Cannot predict (insufficient recent data).")
            return

        last_n_days_scaled = scaler.transform(last_n_days_data)
        last_n_days_reshaped = last_n_days_scaled.reshape(1, lookback_window, scaled_data.shape[1])

        predicted_scaled_price = model.predict(last_n_days_reshaped)

        dummy_array_for_inverse_transform_pred = np.zeros((predicted_scaled_price.shape[0], scaled_data.shape[1]))
        dummy_array_for_inverse_transform_pred[:, 0] = predicted_scaled_price.flatten()
        predicted_price_raw_lstm = scaler.inverse_transform(dummy_array_for_inverse_transform_pred)[:, 0][0]

        last_actual_price = self.data_combined_for_lstm['stock_price'].iloc[-1]
        last_day_sentiment = self.data_combined_for_lstm['sentiment_score'].iloc[-1]
        last_day_headlines = self.data_combined_for_lstm['headlines'].iloc[-1]

        final_predicted_price = predicted_price_raw_lstm
        sentiment_influence_msg = ""
        prediction_date = self.data_combined_for_lstm.index.max() + timedelta(days=1)

        sentiment_threshold = 0.05
        sentiment_adjustment_factor = 0.005

        if last_day_sentiment >= sentiment_threshold:
            sentiment_influence_msg = f"Strong Positive News (score: {last_day_sentiment:.2f}). Adjusting prediction upwards."
            final_predicted_price += abs(predicted_price_raw_lstm * sentiment_adjustment_factor)
        elif last_day_sentiment <= -sentiment_threshold:
            sentiment_influence_msg = f"Strong Negative News (score: {last_day_sentiment:.2f}). Adjusting prediction downwards."
            final_predicted_price -= abs(predicted_price_raw_lstm * sentiment_adjustment_factor)
        else:
            final_predicted_price = last_actual_price + base_trend_daily_change
            sentiment_influence_msg = f"Neutral News (score: {last_day_sentiment:.2f}). Prediction adjusted to follow base trend (avg daily change: ${base_trend_daily_change:.2f})."

        ollama_insight = ""
        try:
            ollama_prompt = f"Analyze the following recent news for {ticker} from {self.data_combined_for_lstm.index.max().strftime('%Y-%m-%d')}:\n\n'{last_day_headlines}'\n\nWhat is the likely impact on the stock price in simple terms (e.g., 'positive impact', 'negative impact', 'neutral impact')? Explain briefly."
            ollama_response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': ollama_prompt}],
                                          stream=False)
            ollama_insight = ollama_response['message']['content']
        except Exception as e:
            ollama_insight = f"Ollama insight failed: {e}. Ensure Ollama server is running and model 'llama3' is pulled."

        self.prediction_label.config(text=f"Prediction for {prediction_date.strftime('%Y-%m-%d')}:\n"
                                          f"Last Actual Close: ${last_actual_price:.2f}\n"
                                          f"Raw LSTM Predicted Close: ${predicted_price_raw_lstm:.2f}\n"
                                          f"Final Adjusted Predicted Close: ${final_predicted_price:.2f}\n"
                                          f"Sentiment Analysis (VADER): {last_day_sentiment:.2f} (Neutral threshold: +/- {sentiment_threshold})\n"
                                          f"Sentiment Influence: {sentiment_influence_msg}\n"
                                          f"\nOllama's News Insight:\n{ollama_insight}")
        messagebox.showinfo("Prediction Complete",
                            "LSTM prediction generated with sentiment influence and Ollama insight.")

    def run(self):
        self.master.mainloop()


# Run the Tkinter app
if __name__ == "__main__":
    root = tk.Tk()
    app = StockNewsPredictorApp(root)
    root.mainloop()
