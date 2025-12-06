import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
from textblob import TextBlob  # Changed from vaderSentiment
from datetime import datetime
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
        master.geometry("1000x700")  # Increased size for better layout

        # Configure grid weights for responsive layout
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(1, weight=1)

        # --- Input Frame ---
        self.input_frame = ttk.LabelFrame(master, text="Stock Ticker & Period")
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(self.input_frame, text="Ticker Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = ttk.Entry(self.input_frame, width=15)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.ticker_entry.insert(0, "ORCL")  # Default value

        ttk.Label(self.input_frame, text="Period (e.g., 1mo, 1y):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.period_entry = ttk.Entry(self.input_frame, width=15)
        self.period_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.period_entry.insert(0, "1mo")  # Default value

        self.fetch_button = ttk.Button(self.input_frame, text="Fetch Data & Analyze",
                                       command=self.start_analysis_thread)
        self.fetch_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.status_label = ttk.Label(self.input_frame, text="Ready")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)

        # --- Plot Frame ---
        self.plot_frame = ttk.LabelFrame(master, text="Historical Stock Price")
        self.plot_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))  # Adjusted size for embedding
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Create a frame for the toolbar to correctly grid it
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # --- News & Sentiment Frame ---
        self.news_sentiment_frame = ttk.LabelFrame(master, text="Recent News & Sentiment")
        self.news_sentiment_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.news_sentiment_frame.grid_rowconfigure(0, weight=1)
        self.news_sentiment_frame.grid_columnconfigure(0, weight=1)

        self.news_tree = ttk.Treeview(self.news_sentiment_frame, columns=("Date", "Title", "Sentiment", "Impact"),
                                      show="headings")
        self.news_tree.heading("Date", text="Date")
        self.news_tree.heading("Title", text="News Headline")
        self.news_tree.heading("Sentiment", text="Sentiment")
        self.news_tree.heading("Impact", text="Impact")

        self.news_tree.column("Date", width=90, anchor="center")
        self.news_tree.column("Title", width=250)
        self.news_tree.column("Sentiment", width=70, anchor="center")
        self.news_tree.column("Impact", width=120, anchor="center")

        self.news_tree_scrollbar_y = ttk.Scrollbar(self.news_sentiment_frame, orient="vertical",
                                                   command=self.news_tree.yview)
        self.news_tree_scrollbar_x = ttk.Scrollbar(self.news_sentiment_frame, orient="horizontal",
                                                   command=self.news_tree.xview)
        self.news_tree.configure(yscrollcommand=self.news_tree_scrollbar_y.set,
                                 xscrollcommand=self.news_tree_scrollbar_x.set)

        self.news_tree.grid(row=0, column=0, sticky="nsew")
        self.news_tree_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.news_tree_scrollbar_x.grid(row=1, column=0, sticky="ew")

        # TextBlob doesn't need to be initialized as an object like VADER's analyzer
        # We will call TextBlob(text).sentiment.polarity directly

    def get_textblob_sentiment(self, text):
        """
        Calculates and returns TextBlob sentiment polarity for a given text.
        Polarity is a float within the range [-1.0, 1.0] where 0 indicates neutral,
        1 indicates positive, and -1 indicates negative.
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def clear_results(self):
        """Clears previous data from the plot and treeview."""
        self.ax.clear()
        self.canvas.draw()
        for item in self.news_tree.get_children():
            self.news_tree.delete(item)

    def start_analysis_thread(self):
        """Starts the data fetching and analysis in a separate thread."""
        ticker_symbol = self.ticker_entry.get().upper()
        period = self.period_entry.get()

        if not ticker_symbol:
            messagebox.showwarning("Input Error", "Please enter a ticker symbol.")
            return

        self.fetch_button.config(state=tk.DISABLED)  # Disable button during processing
        self.status_label.config(text=f"Fetching data for {ticker_symbol}...")

        # Start a new thread for the heavy lifting
        analysis_thread = threading.Thread(target=self._run_analysis, args=(ticker_symbol, period))
        analysis_thread.daemon = True  # Allow the thread to exit with the main app
        analysis_thread.start()

    def _run_analysis(self, ticker_symbol, period):
        """
        Fetches data, performs analysis. Runs in a separate thread.
        Updates GUI via master.after().
        """
        try:
            self.master.after(0, self.clear_results)  # Clear GUI on main thread

            # 1. Get Historical Stock Data
            stock_data = yf.download(ticker_symbol, period=period)
            if stock_data.empty:
                self.master.after(0, messagebox.showinfo, "No Data",
                                  f"No historical data found for {ticker_symbol} for the period {period}.")
                self.master.after(0, self.status_label.config, {"text": "Ready"})
                self.master.after(0, self.fetch_button.config, {"state": tk.NORMAL})
                return

            self.master.after(0, self.plot_historical_data, stock_data, ticker_symbol)

            # 2. Get News Data and Calculate Sentiment
            ticker = yf.Ticker(ticker_symbol)
            news_articles = ticker.news

            sentiment_data_for_gui = []
            if not news_articles:
                self.master.after(0, self.status_label.config,
                                  {"text": f"No recent news found for {ticker_symbol}. Historical data shown."})
            else:
                for article in news_articles:
                    # Correctly access title and pubDate from the 'content' dictionary
                    title = article.get('content', {}).get('title')
                    # 'provider' is a top-level key, 'displayName' is inside it
                    publisher = article.get('provider', {}).get('displayName')
                    publish_time_str = article.get('content', {}).get('pubDate')
                    link = article.get('canonicalUrl', {}).get('url')  # Using canonicalUrl for a more stable link

                    if title and publish_time_str:  # Check if both are present
                        try:
                            # Convert ISO 8601 string to datetime object
                            publish_datetime = datetime.fromisoformat(publish_time_str.replace('Z', '+00:00'))

                            # Use TextBlob for sentiment analysis
                            sentiment_score = self.get_textblob_sentiment(title)

                            impact = "Neutral"
                            if sentiment_score >= 0.05:  # Threshold for positive sentiment
                                impact = "Positive (UP)"
                            elif sentiment_score <= -0.05:  # Threshold for negative sentiment
                                impact = "Negative (DOWN)"

                            sentiment_data_for_gui.append((
                                publish_datetime.strftime('%Y-%m-%d %H:%M'),
                                title,
                                f"{sentiment_score:.3f}",
                                impact
                            ))
                        except ValueError as ve:
                            # Handle cases where pubDate format is unexpected
                            print(f"Skipping news article due to date parsing error: {ve}. Raw article: {article}")
                        except Exception as e:
                            print(f"Skipping news article due to unexpected error: {e}. Raw article: {article}")
                    else:
                        # This 'else' block will now only hit if 'title' or 'pubDate' are genuinely missing
                        # from the 'content' dictionary, or if 'content' itself is missing.
                        print(
                            f"Skipping news article due to missing title or publish time from 'content'. Raw article: {article}")

                # Update the GUI's Treeview from the main thread
                self.master.after(0, self.populate_news_tree, sentiment_data_for_gui)
                self.master.after(0, self.status_label.config, {"text": "Analysis complete!"})

        except Exception as e:
            self.master.after(0, messagebox.showerror, "Error", f"An error occurred: {e}")
            self.master.after(0, self.status_label.config, {"text": "Error occurred."})
        finally:
            self.master.after(0, self.fetch_button.config, {"state": tk.NORMAL})  # Re-enable button

    def plot_historical_data(self, stock_data, ticker_symbol):
        """Updates the Matplotlib plot on the GUI."""
        self.ax.clear()
        self.ax.plot(stock_data.index, stock_data['Close'], label='Close Price')
        self.ax.set_title(
            f'{ticker_symbol} Historical Close Price ({stock_data.index.min().strftime("%Y-%m-%d")} to {stock_data.index.max().strftime("%Y-%m-%d")})')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Price (USD)')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.autofmt_xdate()  # Auto-format dates for better readability
        self.canvas.draw()

    def populate_news_tree(self, sentiment_data):
        """Populates the Treeview with news sentiment data."""
        for item in self.news_tree.get_children():
            self.news_tree.delete(item)  # Clear existing items

        if not sentiment_data:
            self.news_tree.insert("", "end", values=("", "No news with complete data found.", "", ""))
            return

        for data_row in sentiment_data:
            self.news_tree.insert("", "end", values=data_row)


# --- Main application entry point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = StockSentimentApp(root)
    root.mainloop()