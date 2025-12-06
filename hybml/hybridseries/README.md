# Hybrid LLM-LSTM Quantitative Analysis System

## Description
A multi-factor prediction system integrating multivariate deep learning with local Large Language Models (LLMs). This system combines quantitative technical analysis with qualitative fundamental and sentiment analysis to generate price forecasts with confidence intervals.

## Core Features
* **Multivariate LSTM:** Trains on Open, High, Low, Close, Volume, RSI, MACD, and SMA data.
* **LLM Integration:** Connects to local Ollama instances running Llama3 or DeepSeek-R1.
* **Structured Parsing:** Enforces JSON output from the LLM for programmatic usage of qualitative assessments (Macro outlook, Fundamental strength, Risk severity).
* **Risk Management Logic:** quantitatively adjusts price predictions based on specific identified risks (e.g., lawsuits, regulatory issues).
* **Reporting:** Generates interactive HTML reports via Plotly for price predictions and technical indicators.

## Requirements
* **Python:** 3.10+
* **LLM Server:** Ollama running locally.
* **Models:** `deepseek-r1` or `llama3`.
* **Python Libraries:** tensorflow, ollama, plotly, yfinance, pandas, numpy, scikit-learn, vaderSentiment.

## Usage
1.  Ensure Ollama is running (`ollama serve`).
2.  Run the script.
3.  Configure prediction horizon (1-10 days) and lookback window.
4.  Resulting HTML files will open in the default web browser.
