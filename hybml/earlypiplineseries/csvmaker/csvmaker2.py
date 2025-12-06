import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ticker = "AAPL"
print(f"Processing {ticker}...")

# 1. Stock Data
stock_data = yf.download(ticker, period="1y")
stock_data = stock_data[['Close']]

# 2. News Data & Sentiment
analyzer = SentimentIntensityAnalyzer()
ticker_obj = yf.Ticker(ticker)
news = ticker_obj.news

sentiment_data = []
for item in news:
    if 'providerPublishTime' in item:
        date = datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d')
        title = item.get('title', '')
        score = analyzer.polarity_scores(title)['compound']
        sentiment_data.append({'Date': date, 'Sentiment': score})

# 3. Merge
news_df = pd.DataFrame(sentiment_data)
news_df['Date'] = pd.to_datetime(news_df['Date'])
news_df = news_df.groupby('Date').mean() # Average sentiment per day

stock_data.index = pd.to_datetime(stock_data.index).normalize()
combined = stock_data.merge(news_df, left_index=True, right_on='Date', how='left')
combined['Sentiment'] = combined['Sentiment'].fillna(0) # 0 if no news

print(combined.tail())
combined.to_csv("stock_sentiment.csv")
