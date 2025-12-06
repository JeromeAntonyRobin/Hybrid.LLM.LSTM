import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

def get_stock_data(ticker, days=365):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]

def get_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    t = yf.Ticker(ticker)
    data = []
    for item in t.news:
        ts = item.get('providerPublishTime')
        if ts:
            dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            score = analyzer.polarity_scores(item['title'])['compound']
            data.append({'Date': dt, 'Sentiment': score})
    
    if not data:
        return pd.DataFrame(columns=['Date', 'Sentiment'])
        
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.groupby('Date').mean()

def main():
    ticker_input = input("Enter Ticker (e.g., TSLA): ").upper()
    
    print("Fetching data...")
    stocks = get_stock_data(ticker_input)
    news = get_sentiment(ticker_input)
    
    stocks.index = pd.to_datetime(stocks.index).normalize()
    combined = stocks.merge(news, left_index=True, right_on='Date', how='left').fillna(0)
    
    print(f"Data prepared for {ticker_input}. Rows: {len(combined)}")
    # Ready for analysis

if __name__ == "__main__":
    main()
