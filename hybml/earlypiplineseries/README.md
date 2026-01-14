# Stock Data Aggregation and Visualization Tools

## Description
This folder contains Python scripts for retrieving financial data, processing news sentiment, and visualizing correlations between news events and stock price movements. These scripts serve as the exploratory data analysis (EDA) foundation for downstream machine learning tasks.

## Contents
* **csvmaker.py:** ETL pipeline that downloads historical OHLCV data via yfinance, fetches news headlines, applies VADER sentiment scoring, and merges data into a CSV file for model training.
* **sentanalysis Series:** GUI-based tools to visualize stock history.
    * **v1:** Basic price plotting.
    * **v2:** Overlays specific news events on the price chart as color-coded markers based on sentiment polarity.
    * **v3:** Adds calculation of daily percentage changes to quantify the immediate market impact of news.

## Technical Specifications
* **Language:** Python 3.x
* **Libraries:** pandas, yfinance, matplotlib, tkinter, vaderSentiment, textblob
* **Input:** User-defined ticker symbols and date ranges.
* **Output:** CSV datasets and Matplotlib charts embedded in Tkinter GUIs.

## Usage
Run the scripts directly via terminal. Ensure all dependencies are installed.
