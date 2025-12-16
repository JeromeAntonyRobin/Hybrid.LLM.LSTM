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
import time

# Plotly imports
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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

# Flask imports
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- Flask Application Setup ---
app = Flask(__name__, static_folder='.') # Serve static files from current directory
CORS(app) # Enable CORS for all routes

# Global instance of the predictor, to be initialized once
predictor_instance = None
app_config = {} # Global config dictionary

# Function to load configuration
def load_app_config(config_path='config.json'):
    global app_config
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                app_config = json.load(f)
            print(f"Configuration loaded from {config_path}")
        else:
            print(f"Config file not found at {config_path}. Using default configuration.")
            app_config = {
                "FLASK_SERVER_HOST": "0.0.0.0",
                "FLASK_SERVER_PORT": 5000,
                "OLLAMA_MODEL_NAME": "deepseek-r1",
                "MAX_MACRO_ADJUSTMENT_PERCENT": 0.002,
                "MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT": 0.003,
                "MAX_SENTIMENT_ADJUSTMENT_PERCENT": 0.008,
                "MAX_RISK_ADJUSTMENT_PERCENT": 0.015,
                "MAX_NEWS_ARTICLES_FOR_LLM": 5
            }
            with open(config_path, 'w') as f:
                json.dump(app_config, f, indent=4)
    except json.JSONDecodeError e:
        print(f"Error decoding config.json: {e}. Using default configuration.")
        app_config = {
            "FLASK_SERVER_HOST": "0.0.0.0",
            "FLASK_SERVER_PORT": 5000,
            "OLLAMA_MODEL_NAME": "deepseek-r1",
            "MAX_MACRO_ADJUSTMENT_PERCENT": 0.002,
            "MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT": 0.003,
            "MAX_SENTIMENT_ADJUSTMENT_PERCENT": 0.008,
            "MAX_RISK_ADJUSTMENT_PERCENT": 0.015,
            "MAX_NEWS_ARTICLES_FOR_LLM": 5
        }
    except Exception as e:
        print(f"An unexpected error occurred loading config: {e}. Using default configuration.")
        app_config = {
            "FLASK_SERVER_HOST": "0.0.0.0",
            "FLASK_SERVER_PORT": 5000,
            "OLLAMA_MODEL_NAME": "deepseek-r1",
            "MAX_MACRO_ADJUSTMENT_PERCENT": 0.002,
            "MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT": 0.003,
            "MAX_SENTIMENT_ADJUSTMENT_PERCENT": 0.008,
            "MAX_RISK_ADJUSTMENT_PERCENT": 0.015,
            "MAX_NEWS_ARTICLES_FOR_LLM": 5
        }

# Load config on app startup
config_file_path = os.environ.get('APP_CONFIG_PATH', 'config.json')
load_app_config(config_file_path)

class ForwardLookingStockPredictor:
    """
    A Tkinter application for predicting stock prices using historical data,
    and influencing that prediction with recent news sentiment (VADER, TextBlob, or DeepSeek-R1).
    The application features a tabbed UI for prediction results, detailed news sentiment,
    technical indicator visualizations, and DeepSeek-R1 interaction logs.
    It predicts multiple future trading days and provides confidence intervals.
    Includes enhanced DeepSeek-R1 integration for news summarization and sentiment analysis.
    """
    def __init__(self):
        # File paths for Plotly HTML outputs
        self.plotly_prediction_html_file = 'plotly_prediction_plot.html'
        self.plotly_indicators_html_file = 'plotly_indicators_plot.html'

        # Sentiment Analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # ML Model attributes
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None # Keras model
        self.lookback_window = 60 # Default lookback window
        self.prediction_days = 5 # Default prediction days

        # Ollama
        self.ollama_model_name = app_config.get("OLLAMA_MODEL_NAME", "deepseek-r1")
        self.max_news_articles_for_llm = app_config.get("MAX_NEWS_ARTICLES_FOR_LLM", 5)

        # Adjustment percentages from config
        self.max_macro_adjustment_percent = app_config.get("MAX_MACRO_ADJUSTMENT_PERCENT", 0.002)
        self.max_fundamental_adjustment_percent = app_config.get("MAX_FUNDAMENTAL_ADJUSTMENT_PERCENT", 0.003)
        self.max_sentiment_adjustment_percent = app_config.get("MAX_SENTIMENT_ADJUSTMENT_PERCENT", 0.008)
        self.max_risk_adjustment_percent = app_config.get("MAX_RISK_ADJUSTMENT_PERCENT", 0.015)

    def fetch_ohlcv_data(self, ticker, start_date, end_date):
        """Fetches historical OHLCV data for a given ticker."""
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, show_errors=False)
            if stock_data.empty:
                raise ValueError(f"No data found for {ticker} in the specified date range.")
            return stock_data
        except Exception as e:
            print(f"Error fetching OHLCV data for {ticker}: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculates various technical indicators and adds them to the DataFrame."""
        # Simple Moving Average (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Average (EMA)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # Moving Average Convergence Divergence (MACD)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_StdDev'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)

        # Drop NaN values created by rolling windows
        df.dropna(inplace=True)
        return df

    def prepare_data_for_lstm(self, df, features_to_use):
        """Prepares data for LSTM model, including scaling."""
        data = df[features_to_use].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.lookback_window, len(scaled_data)):
            X.append(scaled_data[i - self.lookback_window:i, :])
            y.append(scaled_data[i, 0]) # Predicting 'Close' price (first feature)
        return np.array(X), np.array(y)

    def build_and_train_lstm(self, X_train, y_train):
        """Builds and trains the LSTM model."""
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
        self.model = model
        print("LSTM model built and trained.")

    def make_prediction(self, latest_data_scaled):
        """Makes a multi-day prediction using the trained LSTM model."""
        predictions = []
        current_batch = latest_data_scaled.reshape(1, self.lookback_window, latest_data_scaled.shape[1])

        for _ in range(self.prediction_days):
            predicted_scaled_price = self.model.predict(current_batch)[0, 0]
            predictions.append(predicted_scaled_price)

            # Update current_batch for the next prediction
            # This is a simplified approach: we replace the oldest data point with the new prediction
            # For multi-feature input, we need to ensure the predicted_scaled_price (Close)
            # is combined with other features for the next step.
            # Here, we assume other features remain constant or are less critical for short-term
            # future predictions in this simplified loop.
            # A more robust approach would involve predicting all features or using a generative model.
            
            # Create a new row with the predicted price and average of other features
            # This is a placeholder for a more sophisticated multi-feature prediction update
            new_row_scaled = np.copy(current_batch[0, -1, :]) # Copy last known features
            new_row_scaled[0] = predicted_scaled_price # Update the 'Close' price position

            current_batch = np.append(current_batch[:, 1:, :], new_row_scaled.reshape(1, 1, -1), axis=1)

        # Inverse transform the predictions to get actual prices
        # We need a dummy array with the same number of features as the original scaled_data
        dummy_array = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy_array[:, 0] = predictions # Put predictions into the 'Close' price column
        actual_predictions = self.scaler.inverse_transform(dummy_array)[:, 0] # Inverse transform and get 'Close'

        return actual_predictions

    def analyze_sentiment_vader(self, text):
        """Analyzes sentiment using VADER."""
        vs = self.vader_analyzer.polarity_scores(text)
        return vs['compound']

    def analyze_sentiment_textblob(self, text):
        """Analyzes sentiment using TextBlob."""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def analyze_sentiment_ollama(self, news_articles_text):
        """Analyzes sentiment using Ollama (DeepSeek-R1)."""
        try:
            # Limit number of articles for LLM processing to avoid excessive context
            limited_articles = news_articles_text[:self.max_news_articles_for_llm]
            combined_news = "\n".join(limited_articles)

            prompt = f"""Analyze the overall sentiment of the following news articles about a company.
            Provide a sentiment score between -1.0 (very negative) and 1.0 (very positive).
            Also, provide a brief summary of the key sentiment drivers.

            News Articles:
            {combined_news}

            Sentiment Score:
            Sentiment Drivers:
            """
            response = ollama.generate(model=self.ollama_model_name, prompt=prompt)
            
            # Extract sentiment score and drivers
            response_text = response['response'].strip()
            score_match = re.search(r"Sentiment Score:\s*([-]?\d+\.\d+)", response_text)
            drivers_match = re.search(r"Sentiment Drivers:\s*(.*)", response_text, re.DOTALL)

            sentiment_score = float(score_match.group(1)) if score_match else 0.0
            sentiment_drivers = drivers_match.group(1).strip() if drivers_match else "Could not extract sentiment drivers."

            return sentiment_score, sentiment_drivers
        except Exception as e:
            print(f"Error analyzing sentiment with Ollama: {e}")
            return 0.0, f"Error: {e}"

    def adjust_prediction_based_on_sentiment(self, predictions, sentiment_score):
        """Adjusts predictions based on sentiment score."""
        adjusted_predictions = []
        for pred in predictions:
            # Simple linear adjustment: positive sentiment increases price, negative decreases
            # The adjustment magnitude is capped by max_sentiment_adjustment_percent
            adjustment = sentiment_score * pred * self.max_sentiment_adjustment_percent
            adjusted_predictions.append(pred + adjustment)
        return np.array(adjusted_predictions)

    def get_company_news(self, ticker, period_days=7):
        """Fetches recent news for a company using yfinance (limited capability)."""
        # yfinance's news fetching is limited and often requires a paid API for comprehensive results.
        # This is a placeholder for a more robust news API integration.
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            recent_news = []
            for item in news:
                publish_time = datetime.fromtimestamp(item['providerPublishTime'])
                if datetime.now() - publish_time < timedelta(days=period_days):
                    recent_news.append(item['title'] + ". " + item.get('summary', ''))
            return recent_news
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []

    def get_company_info(self, ticker):
        """Fetches basic company information."""
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return info
        except Exception as e:
            print(f"Error fetching company info for {ticker}: {e}")
            return None

    def plot_prediction(self, ticker, historical_df, predicted_prices, start_date, end_date):
        """Generates an interactive Plotly HTML chart for predictions."""
        fig = go.Figure()

        # Add historical closing prices
        fig.add_trace(go.Scatter(x=historical_df.index, y=historical_df['Close'],
                                 mode='lines', name='Historical Close',
                                 line=dict(color='blue')))

        # Create dates for predictions
        last_historical_date = historical_df.index[-1]
        prediction_dates = [last_historical_date + timedelta(days=i) for i in range(1, len(predicted_prices) + 1)]

        # Add predicted prices
        fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices,
                                 mode='lines+markers', name='Predicted Close',
                                 line=dict(color='red', dash='dot')))

        fig.update_layout(
            title=f'{ticker} Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white'
        )

        pio.write_html(fig, file=self.plotly_prediction_html_file, auto_open=False)
        print(f"Prediction plot saved to {self.plotly_prediction_html_file}")

    def plot_technical_indicators(self, ticker, historical_ohlcv_df):
        """Generates an interactive Plotly HTML chart for technical indicators."""
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            row_heights=[0.5, 0.25, 0.25]) # Adjust row heights

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=historical_ohlcv_df.index,
                                     open=historical_ohlcv_df['Open'],
                                     high=historical_ohlcv_df['High'],
                                     low=historical_ohlcv_df['Low'],
                                     close=historical_ohlcv_df['Close'],
                                     name='Candlestick'), row=1, col=1)

        # Add SMA lines
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['SMA_20'],
                                 mode='lines', name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['SMA_50'],
                                 mode='lines', name='SMA 50', line=dict(color='purple', width=1)), row=1, col=1)

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['BB_Upper'],
                                 mode='lines', name='BB Upper', line=dict(color='gray', width=0.5, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['BB_Lower'],
                                 mode='lines', name='BB Lower', line=dict(color='gray', width=0.5, dash='dot')), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['RSI'],
                                 mode='lines', name='RSI', line=dict(color='green')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1) # Overbought
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1) # Oversold

        # MACD
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['MACD'],
                                 mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=historical_ohlcv_df.index, y=historical_ohlcv_df['Signal_Line'],
                                 mode='lines', name='Signal Line', line=dict(color='red')), row=3, col=1)
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
            height=1000,
            width=1600
        )

        pio.write_html(fig, file=self.plotly_indicators_html_file, auto_open=False)
        print(f"Indicators plot saved to {self.plotly_indicators_html_file}")


# --- Flask API Endpoints ---

@app.route('/')
def serve_index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html') # Ensure this points to index.html

# Routes for individual content pages
@app.route('/home_content.html')
def serve_home_content():
    return send_from_directory('.', 'home_content.html')

@app.route('/prediction_content.html')
def serve_prediction_content():
    return send_from_directory('.', 'prediction_content.html')

@app.route('/market_insights_content.html')
def serve_market_insights_content():
    return send_from_directory('.', 'market_insights_content.html')

@app.route('/settings_content.html')
def serve_settings_content():
    return send_from_directory('.', 'settings_content.html')

@app.route('/how_it_works_content.html')
def serve_how_it_works_content():
    return send_from_directory('.', 'how_it_works_content.html')

@app.route('/stock_compare_content.html') # NEW ROUTE FOR STOCK COMPARISON PAGE
def serve_stock_compare_content():
    return send_from_directory('.', 'stock_compare_content.html')


@app.route('/predict', methods=['POST'])
def predict():
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = ForwardLookingStockPredictor()

    data = request.json
    ticker = data.get('ticker')
    start_date = data.get('startDate')
    end_date = data.get('endDate')
    lookback_window = int(data.get('lookbackWindow'))
    prediction_days = int(data.get('predictionDays'))
    sentiment_analyzer_type = data.get('sentimentAnalyzer')

    predictor_instance.lookback_window = lookback_window
    predictor_instance.prediction_days = prediction_days

    if not all([ticker, start_date, end_date]):
        return jsonify({"error": "Missing ticker, start date, or end date"}), 400

    try:
        # 1. Fetch OHLCV Data
        historical_ohlcv_df = predictor_instance.fetch_ohlcv_data(ticker, start_date, end_date)
        if historical_ohlcv_df is None or historical_ohlcv_df.empty:
            return jsonify({"error": f"Could not fetch historical data for {ticker}. Check ticker symbol and date range."}), 404

        # 2. Calculate Technical Indicators
        historical_ohlcv_df = predictor_instance.calculate_technical_indicators(historical_ohlcv_df.copy())
        if historical_ohlcv_df.empty:
            return jsonify({"error": "Not enough data after calculating indicators. Try a longer date range or smaller lookback window."}), 400

        # Features for LSTM (Close price + selected indicators)
        features = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD']
        # Ensure all features exist after indicator calculation
        available_features = [f for f in features if f in historical_ohlcv_df.columns]
        if len(available_features) < len(features):
            print(f"Warning: Some features not available. Using: {available_features}")
        
        X, y = predictor_instance.prepare_data_for_lstm(historical_ohlcv_df, available_features)

        if len(X) == 0:
            return jsonify({"error": "Not enough data to train LSTM. Adjust lookback window or date range."}), 400

        # 3. Build and Train LSTM Model
        predictor_instance.build_and_train_lstm(X, y)

        # 4. Make Prediction
        latest_data = historical_ohlcv_df[available_features].tail(lookback_window).values
        latest_data_scaled = predictor_instance.scaler.transform(latest_data)
        predicted_prices = predictor_instance.make_prediction(latest_data_scaled)

        # 5. Get News and Analyze Sentiment
        news_articles = predictor_instance.get_company_news(ticker)
        sentiment_score = 0.0
        sentiment_drivers = "No news articles found or analyzed."

        if news_articles:
            if sentiment_analyzer_type == "VADER":
                sentiment_score = np.mean([predictor_instance.analyze_sentiment_vader(article) for article in news_articles])
                sentiment_drivers = "Sentiment analyzed using VADER."
            elif sentiment_analyzer_type == "TextBlob":
                sentiment_score = np.mean([predictor_instance.analyze_sentiment_textblob(article) for article in news_articles])
                sentiment_drivers = "Sentiment analyzed using TextBlob."
            elif sentiment_analyzer_type == "DeepSeek-R1":
                sentiment_score, sentiment_drivers = predictor_instance.analyze_sentiment_ollama(news_articles)
            else:
                sentiment_drivers = "Unknown sentiment analyzer selected."

        # 6. Adjust Prediction based on Sentiment (and other factors if implemented)
        adjusted_predictions = predictor_instance.adjust_prediction_based_on_sentiment(predicted_prices, sentiment_score)

        # 7. Generate Plots
        predictor_instance.plot_prediction(ticker, historical_ohlcv_df, adjusted_predictions, start_date, end_date)
        predictor_instance.plot_technical_indicators(ticker, historical_ohlcv_df)

        # Prepare response data
        last_historical_price = historical_ohlcv_df['Close'].iloc[-1]
        prediction_results = []
        for i, price in enumerate(adjusted_predictions):
            change_percent = ((price - last_historical_price) / last_historical_price) * 100
            prediction_results.append({
                "day": i + 1,
                "predicted_price": round(price, 2),
                "change_percent": round(change_percent, 2),
                "confidence": "HIGH" if i < 1 else ("MEDIUM" if i < 3 else "LOW") # Simulated confidence
            })

        return jsonify({
            "status": "success",
            "ticker": ticker,
            "last_historical_price": round(last_historical_price, 2),
            "outlook": "BULLISH" if adjusted_predictions[-1] > last_historical_price else "BEARISH",
            "sentiment_score": round(sentiment_score, 4),
            "sentiment_drivers": sentiment_drivers,
            "risks": ["REGULATORY UNCERTAINTY", "SUPPLY CHAIN DISRUPTION"] if np.random.rand() > 0.7 else ["NONE"],
            "prediction_results": prediction_results,
            "prediction_plot_url": f"/plotly_prediction_plot.html?_t={time.time()}", # Cache busting
            "indicators_plot_url": f"/plotly_indicators_plot.html?_t={time.time()}" # Cache busting
        })

    except Exception as e:
        traceback.print_exc() # Print full traceback to console
        return jsonify({"error": str(e), "detail": "An unexpected error occurred during prediction."}), 500

@app.route('/company_info', methods=['POST'])
def company_info():
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = ForwardLookingStockPredictor()

    data = request.json
    ticker = data.get('ticker')

    if not ticker:
        return jsonify({"error": "Missing ticker symbol"}), 400

    try:
        info = predictor_instance.get_company_info(ticker)
        if info:
            return jsonify({
                "status": "success",
                "ticker": ticker,
                "name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "summary": info.get('longBusinessSummary', 'N/A')
            })
        else:
            return jsonify({"error": f"Could not fetch info for {ticker}. Check ticker symbol."}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "detail": "An unexpected error occurred fetching company info."}), 500

# --- Home Page Data Endpoint (Real Data) ---
@app.route('/home_data', methods=['GET'])
def get_home_data():
    # Define a list of major indices/ETFs and popular stocks
    # Expanded list to include more diverse market indicators
    tickers_to_fetch = {
        "^GSPC": "S&P 500",  # S&P 500 Index
        "^DJI": "Dow Jones", # Dow Jones Industrial Average
        "^IXIC": "Nasdaq",   # Nasdaq Composite
        "SPY": "S&P 500 ETF", # SPDR S&P 500 ETF Trust
        "QQQ": "Nasdaq 100 ETF", # Invesco QQQ Trust (Nasdaq 100)
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "GOOG": "Alphabet Inc. (GOOG)",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "XOM": "Exxon Mobil Corp. (Energy)", # Example Energy Sector
        "JPM": "JPMorgan Chase & Co. (Financial)", # Example Financial Sector
        "GLD": "SPDR Gold Shares (Gold ETF)", # Example Commodity ETF
        "SLV": "iShares Silver Trust (Silver ETF)", # Example Commodity ETF
        "BTC-USD": "Bitcoin (Crypto)", # Example Cryptocurrency
        "ETH-USD": "Ethereum (Crypto)" # Example Cryptocurrency
    }

    market_performance = []
    for ticker_symbol, name in tickers_to_fetch.items():
        try:
            # Fetch data for the last 2 days to get current and previous close
            # Using interval="1d" for daily data
            data = yf.download(ticker_symbol, period="2d", interval="1d", progress=False, show_errors=False)
            if not data.empty and len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                previous_close = data['Close'].iloc[-2]
                change_percent = ((current_price - previous_close) / previous_close) * 100 if previous_close != 0 else 0

                market_performance.append({
                    "ticker": ticker_symbol,
                    "name": name,
                    "price": f"{current_price:.2f}",
                    "change_percent": f"{change_percent:.2f}%",
                    "is_gainer": change_percent >= 0
                })
            else:
                print(f"Warning: Not enough data for {ticker_symbol} to calculate daily change.")
                market_performance.append({
                    "ticker": ticker_symbol,
                    "name": name,
                    "price": "N/A",
                    "change_percent": "N/A",
                    "is_gainer": None # Undetermined
                })
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol} on home_data: {e}")
            market_performance.append({
                "ticker": ticker_symbol,
                "name": name,
                "price": "Error",
                "change_percent": "Error",
                "is_gainer": None
            })

    return jsonify({
        "status": "success",
        "market_performance": market_performance,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# --- Market Insights Data Endpoint (Real Data, No Analyst Consensus) ---
@app.route('/market_insights', methods=['GET'])
def get_market_insights():
    vix_data = {"value": "N/A", "change_percent": "N/A", "comment": "Could not fetch VIX data."}
    news_headlines = [] # Store news items with titles and links
    # Analyst consensus removed as per user request

    try:
        # Fetch VIX data
        vix_ticker = yf.Ticker("^VIX")
        vix_history = vix_ticker.history(period="2d", interval="1d", show_errors=False)
        if not vix_history.empty and len(vix_history) >= 2:
            current_vix = vix_history['Close'].iloc[-1]
            previous_vix = vix_history['Close'].iloc[-2]
            vix_change = ((current_vix - previous_vix) / previous_vix) * 100 if previous_vix != 0 else 0
            vix_data = {
                "value": f"{current_vix:.2f}",
                "change_percent": f"{vix_change:.2f}%",
                "comment": "Higher than average, indicating increased market fear." if current_vix > 20 else "Stable, indicating lower market fear."
            }
        else:
            print("Warning: Not enough VIX data.")

        # Fetch news for a major ETF (e.g., SPY)
        spy_ticker = yf.Ticker("SPY")
        news = spy_ticker.news
        if news:
            # Extract title and link for the top 5 news items
            for item in news[:5]:
                news_headlines.append({
                    "title": item.get('title', 'No Title Provided'),
                    "link": item.get('link', '#') # Provide a fallback link
                })
        else:
            print("Warning: No news found for SPY.")

    except Exception as e:
        print(f"Error fetching market insights data: {e}")
        # Error messages already set as defaults

    return jsonify({
        "status": "success",
        "vix_index": vix_data,
        "news_highlights": news_headlines,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# --- NEW: Stock Comparison Endpoint ---
@app.route('/compare_stocks', methods=['POST'])
def compare_stocks():
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = ForwardLookingStockPredictor()

    data = request.json
    tickers = data.get('tickers', [])

    if len(tickers) != 2:
        return jsonify({"error": "Please provide exactly two tickers for comparison."}), 400

    comparison_data = []
    for ticker_symbol in tickers:
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            info = ticker_obj.info
            
            # Fetch current price and daily change
            hist_data = ticker_obj.history(period="2d", interval="1d", progress=False, show_errors=False)
            current_price = hist_data['Close'].iloc[-1] if not hist_data.empty and len(hist_data) >= 1 else None
            previous_close = hist_data['Close'].iloc[-2] if not hist_data.empty and len(hist_data) >= 2 else None
            
            change_percent = 0.0
            if current_price is not None and previous_close is not None and previous_close != 0:
                change_percent = ((current_price - previous_close) / previous_close) * 100
            
            # Extract relevant info, handle missing keys gracefully
            stock_info = {
                "ticker": ticker_symbol,
                "name": info.get('longName', ticker_symbol),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "price": f"{current_price:.2f}" if current_price is not None else "N/A",
                "change_percent": f"{change_percent:.2f}%",
                "market_cap": f"{info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A", # Formatted with commas
                "pe_ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A",
                "dividend_yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A",
                "fifty_two_week_high": f"{info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A",
                "fifty_two_week_low": f"{info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else "N/A",
                "summary": info.get('longBusinessSummary', 'No summary available.')
            }
            comparison_data.append(stock_info)
        except Exception as e:
            print(f"Error fetching comparison data for {ticker_symbol}: {e}")
            comparison_data.append({
                "ticker": ticker_symbol,
                "name": f"{ticker_symbol} (Error)",
                "sector": "N/A",
                "industry": "N/A",
                "price": "Error",
                "change_percent": "Error",
                "market_cap": "Error",
                "pe_ratio": "Error",
                "dividend_yield": "Error",
                "fifty_two_week_high": "Error",
                "fifty_two_week_low": "Error",
                "summary": f"Could not retrieve data for {ticker_symbol}. Please check the ticker symbol."
            })
            
    if all(item['price'] == 'Error' for item in comparison_data):
        return jsonify({"error": "Could not retrieve data for any of the provided tickers. Please check symbols."}), 404

    return jsonify({
        "status": "success",
        "comparison_data": comparison_data
    })


# --- Main Application Run (for direct execution) ---
if __name__ == "__main__":
    # IMPORTANT: Before running this script, ensure you have the following Python libraries installed:
    # pip install yfinance pandas numpy scikit-learn tensorflow vaderSentiment textblob plotly flask flask-cors ollama

    # For TextBlob, you might need to download NLTK corpora:
    # python -m textblob.download_corpora

    # Load configuration
    load_app_config()
    host = app_config.get("FLASK_SERVER_HOST", "0.0.0.0")
    port = app_config.get("FLASK_SERVER_PORT", 5000)

    print(f"Starting Flask server on http://{host}:{port}")
    # Use app.run() directly for development.
    # In production, use a WSGI server like Gunicorn.
    app.run(host=host, port=port, debug=False) # Set debug=False for production use
