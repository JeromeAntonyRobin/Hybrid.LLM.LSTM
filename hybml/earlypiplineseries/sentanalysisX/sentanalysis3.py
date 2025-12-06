import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import os

# Suppress TensorFlow warnings (optional, if you had it in your environment)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class StockSentimentApp:
    def __init__(self, master):
        self.master = master
        master.title("Stock Price & News Sentiment Analyzer")
        master.geometry("1400x800")  # Adjusted height as AI sections are removed

        # Configure grid weights for responsive layout
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(1, weight=1)

        # --- Input Frame ---
        self.input_frame = ttk.LabelFrame(master, text="Stock Ticker & Date Range")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.input_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = ttk.Entry(self.input_frame, width=15)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.ticker_entry.insert(0, "ORCL")  # Default value

        ttk.Label(self.input_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = ttk.Entry(self.input_frame, width=15)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, (datetime.now() - timedelta(days=365 * 2)).strftime(
            '%Y-%m-%d'))  # Default 2 years ago

        ttk.Label(self.input_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = ttk.Entry(self.input_frame, width=15)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))  # Default today

        self.fetch_button = ttk.Button(self.input_frame, text="Fetch Data & Analyze",
                                       command=self.start_analysis_thread)
        self.fetch_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.status_label = ttk.Label(self.input_frame, text="Ready")
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5)

        # --- Sentiment Analyzer Options Frame ---
        self.sentiment_options_frame = ttk.LabelFrame(self.input_frame, text="Sentiment Analyzer Options")
        self.sentiment_options_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.sentiment_analyzer_var = tk.StringVar(value="TextBlob")  # Default to TextBlob

        self.textblob_radio = ttk.Radiobutton(self.sentiment_options_frame, text="TextBlob",
                                              variable=self.sentiment_analyzer_var, value="TextBlob")
        self.textblob_radio.grid(row=0, column=0, padx=5, pady=2, sticky="w")

        self.vader_radio = ttk.Radiobutton(self.sentiment_options_frame, text="VADER",
                                           variable=self.sentiment_analyzer_var, value="VADER")
        self.vader_radio.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        self.vader_analyzer = SentimentIntensityAnalyzer()  # Initialize VADER analyzer

        # --- Plot Frame ---
        self.plot_frame = ttk.LabelFrame(master, text="Historical Stock Price with News Events")
        self.plot_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))  # Adjusted size for embedding
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # --- News & Sentiment Frame ---
        self.news_sentiment_frame = ttk.LabelFrame(master, text="Recent News & Sentiment (from yfinance)")
        self.news_sentiment_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.news_sentiment_frame.grid_rowconfigure(0, weight=1)
        self.news_sentiment_frame.grid_columnconfigure(0, weight=1)

        # Added "Daily Change %" column
        self.news_tree = ttk.Treeview(self.news_sentiment_frame,
                                      columns=("Date", "Headline", "Description", "Sentiment", "Impact",
                                               "Daily Change %"), show="headings")
        self.news_tree.heading("Date", text="Date")
        self.news_tree.heading("Headline", text="News Headline")
        self.news_tree.heading("Description", text="Description")
        self.news_tree.heading("Sentiment", text="Sentiment")
        self.news_tree.heading("Impact", text="Impact")
        self.news_tree.heading("Daily Change %", text="Daily Change %")  # New heading

        self.news_tree.column("Date", width=90, anchor="center")
        self.news_tree.column("Headline", width=200, minwidth=150)
        self.news_tree.column("Description", width=300, minwidth=200)
        self.news_tree.column("Sentiment", width=70, anchor="center")
        self.news_tree.column("Impact", width=100, anchor="center")
        self.news_tree.column("Daily Change %", width=90, anchor="center")

        self.news_tree_scrollbar_y = ttk.Scrollbar(self.news_sentiment_frame, orient="vertical",
                                                   command=self.news_tree.yview)
        self.news_tree_scrollbar_x = ttk.Scrollbar(self.news_sentiment_frame, orient="horizontal",
                                                   command=self.news_tree.xview)
        self.news_tree.configure(yscrollcommand=self.news_tree_scrollbar_y.set,
                                 xscrollcommand=self.news_tree_scrollbar_x.set)

        self.news_tree.grid(row=0, column=0, sticky="nsew")
        self.news_tree_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.news_tree_scrollbar_x.grid(row=1, column=0, sticky="ew")

        # Removed the bind for AI analysis as it's no longer present
        # self.news_tree.bind("<<TreeviewSelect>>", self.on_news_select)

    def get_vader_sentiment(self, text):
        """Calculates and returns VADER sentiment compound score."""
        sentiment_scores = self.vader_analyzer.polarity_scores(text)
        return sentiment_scores['compound']

    def get_textblob_sentiment(self, text):
        """Calculates and returns TextBlob sentiment polarity."""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def calculate_model_sentiment(self, text):
        """Calculates sentiment using the currently selected model (TextBlob or VADER)."""
        selected_analyzer = self.sentiment_analyzer_var.get()
        if selected_analyzer == "VADER":
            return self.get_vader_sentiment(text)
        else:  # Default to TextBlob
            return self.get_textblob_sentiment(text)

    def get_impact_label(self, sentiment_score):
        """Determines impact label based on sentiment score."""
        if sentiment_score >= 0.05:
            return "Positive (UP)"
        elif sentiment_score <= -0.05:
            return "Negative (DOWN)"
        else:
            return "Neutral"

    def clear_results(self):
        """Clears previous data from the plot and news treeview."""
        self.ax.clear()
        self.canvas.draw()
        for item in self.news_tree.get_children():
            self.news_tree.delete(item)

    def start_analysis_thread(self):
        """Starts the yfinance data fetching and analysis in a separate thread."""
        ticker_symbol = self.ticker_entry.get().upper()
        start_date_str = self.start_date_entry.get()
        end_date_str = self.end_date_entry.get()

        if not ticker_symbol:
            messagebox.showwarning("Input Error", "Please enter a ticker symbol.")
            return

        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            if start_date > end_date:
                messagebox.showwarning("Input Error", "Start Date cannot be after End Date.")
                return
        except ValueError:
            messagebox.showwarning("Input Error", "Please enter dates in YYYY-MM-DD format.")
            return

        self.fetch_button.config(state=tk.DISABLED)  # Disable button during processing
        self.status_label.config(
            text=f"Fetching yfinance data for {ticker_symbol} from {start_date_str} to {end_date_str}...")

        analysis_thread = threading.Thread(target=self._run_yfinance_analysis,
                                           args=(ticker_symbol, start_date_str, end_date_str))
        analysis_thread.daemon = True
        analysis_thread.start()

    def _run_yfinance_analysis(self, ticker_symbol, start_date_str, end_date_str):
        """
        Fetches data and performs analysis. Runs in a separate thread.
        Updates GUI via master.after().
        """
        try:
            self.master.after(0, self.clear_results)  # Clear GUI on main thread

            # Fetch historical stock data using start and end dates
            stock_data = yf.download(ticker_symbol, start=start_date_str, end=end_date_str)
            if stock_data.empty:
                self.master.after(0, messagebox.showinfo, "No Data",
                                  f"No historical data found for {ticker_symbol} from {start_date_str} to {end_date_str}.")
                self.master.after(0, self.status_label.config, {"text": "Ready"})
                self.master.after(0, self.fetch_button.config, {"state": tk.NORMAL})
                return

            # Calculate daily percentage change for stock data
            # Ensure stock_data is sorted by index (date) for pct_change to work correctly
            stock_data = stock_data.sort_index()
            stock_data['Daily Change %'] = stock_data['Close'].pct_change() * 100

            # Get News Data (still limited to recent news by yfinance)
            ticker = yf.Ticker(ticker_symbol)
            news_articles = ticker.news

            sentiment_data_for_gui = []
            news_events_for_plot = []

            if not news_articles:
                self.master.after(0, self.status_label.config, {
                    "text": f"No recent news found for {ticker_symbol} from yfinance. Historical price data shown."})
            else:
                for article in news_articles:
                    title = article.get('content', {}).get('title')
                    description = article.get('content', {}).get('summary') or \
                                  article.get('content', {}).get('description') or ""
                    publish_time_str = article.get('content', {}).get('pubDate')

                    if title and publish_time_str:
                        try:
                            publish_datetime = datetime.fromisoformat(publish_time_str.replace('Z', '+00:00'))
                            news_date = publish_datetime.date()

                            # Get daily change for the news date if available in stock_data
                            daily_change_percent_str = "N/A"
                            if not stock_data.empty:
                                # Find the closest trading day in stock_data for the news event
                                # Convert news_date to Timestamp for comparison
                                news_date_ts = pd.Timestamp(news_date)

                                if news_date_ts in stock_data.index:
                                    daily_change_percent = stock_data.loc[news_date_ts, 'Daily Change %']
                                    daily_change_percent_str = f"{daily_change_percent:.2f}%"
                                else:
                                    # If news date is not a trading day, try finding the next closest trading day
                                    # or previous, for display purposes. We'll stick to the date the news came out for plotting
                                    # but for the table, linking to the exact day's stock price might not be possible if it's a weekend/holiday
                                    pass  # Keep N/A if exact match not found for simplicity in table

                            sentiment_score = self.calculate_model_sentiment(title)
                            impact = self.get_impact_label(sentiment_score)

                            sentiment_data_for_gui.append((
                                publish_datetime.strftime('%Y-%m-%d %H:%M'),
                                title,
                                description,
                                f"{sentiment_score:.3f}",
                                impact,
                                daily_change_percent_str
                            ))

                            news_events_for_plot.append({
                                'date': news_date,
                                'sentiment': sentiment_score,
                                'color': "green" if sentiment_score >= 0.05 else (
                                    "red" if sentiment_score <= -0.05 else "blue")
                            })

                        except ValueError as ve:
                            print(
                                f"Skipping yfinance news article due to date parsing error: {ve}. Raw article: {article}")
                        except Exception as e:
                            print(
                                f"Skipping yfinance news article due to unexpected error: {e}. Raw article: {article}")
                    else:
                        print(
                            f"Skipping yfinance news article due to missing headline/description/publish time. Raw article: {article}")

                self.master.after(0, self.plot_historical_data, stock_data, ticker_symbol, news_events_for_plot)
                self.master.after(0, self.populate_news_tree, sentiment_data_for_gui)
                self.master.after(0, self.status_label.config, {"text": "yfinance analysis complete!"})

        except Exception as e:
            self.master.after(0, messagebox.showerror, "Error", f"An error occurred during yfinance analysis: {e}")
            self.master.after(0, self.status_label.config, {"text": "yfinance analysis error."})
        finally:
            self.master.after(0, self.fetch_button.config, {"state": tk.NORMAL})

    def plot_historical_data(self, stock_data, ticker_symbol, news_events):
        """Updates the Matplotlib plot on the GUI, including news event markers."""
        self.ax.clear()
        self.ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='black')
        self.ax.set_title(f'{ticker_symbol} Historical Close Price & News Events')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price (USD)')
        self.ax.legend()
        self.ax.grid(True)

        for event in news_events:
            event_date = event['date']
            # Find the closest trading day in stock_data for the news event
            if not stock_data.empty:
                closest_stock_date = None
                # Check if news_date is directly in the stock_data index
                if pd.Timestamp(event_date) in stock_data.index:
                    closest_stock_date = pd.Timestamp(event_date)
                else:
                    # Find the nearest valid trading day. Using .get_indexer with 'nearest' method.
                    # Convert event_date to Timestamp for get_indexer
                    event_date_ts = pd.Timestamp(event_date)

                    # Ensure stock_data index is datetime-like for get_indexer
                    if not isinstance(stock_data.index, pd.DatetimeIndex):
                        stock_data.index = pd.to_datetime(stock_data.index)

                    loc = stock_data.index.get_indexer([event_date_ts], method='nearest')[0]
                    # Check if the closest date is within a reasonable proximity (e.g., +/- 3 days)
                    # This prevents linking news from far away to a stock price.
                    if abs((stock_data.index[loc] - event_date_ts).days) <= 3:
                        closest_stock_date = stock_data.index[loc]

                if closest_stock_date:
                    price_at_event = stock_data.loc[closest_stock_date]['Close']
                    self.ax.plot(closest_stock_date, price_at_event, marker='o', markersize=8, color=event['color'],
                                 alpha=0.7,
                                 label=f"News Event ({event['color'].capitalize()})")
                # else:
                #     print(f"No sufficiently close trading day found for news event on {event_date}.")
            # else:
            #     print(f"Stock data is empty, cannot plot news event for date {event_date}.")

        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if "News Event" in label and label in unique_labels:
                continue
            unique_labels[label] = handle
        self.ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')

        self.fig.autofmt_xdate()
        self.canvas.draw()

    def populate_news_tree(self, sentiment_data_for_gui):
        """Populates the Treeview with yfinance news sentiment data."""
        for item in self.news_tree.get_children():
            self.news_tree.delete(item)

        if not sentiment_data_for_gui:
            self.news_tree.insert("", "end", values=("", "No news found.", "", "", "", ""))
            return

        for i, data_row in enumerate(sentiment_data_for_gui):
            # The last element of data_row (full_article_content) is no longer needed
            # as AI analysis is removed and this data wasn't explicitly used for display.
            self.news_tree.insert("", "end", iid=str(i), values=data_row)


# --- Main application entry point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = StockSentimentApp(root)
    root.mainloop()
