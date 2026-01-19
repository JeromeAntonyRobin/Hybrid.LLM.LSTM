# Hybrid Engine: Latest Stable Prototype
## This directory contains the latest stable build of this series of Hybrid Intelligence Stock Prediction Engine. This version features a fully threaded Tkinter GUI, integrated DeepSeek-R1 reasoning via Ollama, and interactive Plotly visualizations.

## Prerequisites
Before running the engine, ensure your system meets the following requirements:

### 1. System Requirements
OS: Windows 10/11 (Recommended for WSL/Ollama support), macOS, or Linux.

Python: Version 3.9 or higher.

RAM: Minimum 16GB (Running local LLMs alongside LSTMs is memory intensive).

GPU: NVIDIA GPU recommended for faster Ollama inference (optional but preferred).

### 2. Local LLM Setup (Ollama)
This engine relies on Ollama to run the DeepSeek-R1 model locally.

Download and install Ollama from ollama.com.

Open your terminal/command prompt and pull the reasoning model:

Bash
```
ollama pull deepseek-r1
```
Ensure the Ollama server is running in the background before starting the app.

🛠️ Installation
Navigate to this directory:

Bash
```
cd hybml/hybridseries/finalthatworks
#Install Python Dependencies: We recommend using a virtual environment. I used Anaconda Conda Environment.


pip install -r requirements.txt



pip install yfinance pandas numpy scikit-learn tensorflow vaderSentiment textblob plotly ollama

```
🚀 Usage Guide
Launch the Application: Run the main application script:

Bash
```
python lazyapproachfinal.py
```
---

## Configure Prediction Parameters:

Ticker: Enter a valid stock symbol (e.g., IBM, AAPL, NVDA).

Dates: Select your historical training window (recommended: 2 years of data).

Lookback: The number of past days the LSTM considers for one prediction (default: 60).

Analyzer: Select DeepSeek-R1 for the full hybrid reasoning experience.
## Adjust Parameters!
<img width="1356" height="717" alt="image" src="https://github.com/user-attachments/assets/c88b2604-515e-4604-96be-9dde1da5ad24" />




---

## Run Prediction:

### Click "Predict Prices & Generate Plots".

<img width="1346" height="693" alt="Screenshot 2026-01-19 232014" src="https://github.com/user-attachments/assets/6d149694-aaa9-433f-987c-b1ac93b9705a" />

### Wait: The system will first fetch data, then "think" (querying the local LLM for sentiment/risk), and finally train the LSTM. This process may take 1-3 minutes depending on your hardware.

<img width="1345" height="675" alt="Screenshot 2026-01-19 232145" src="https://github.com/user-attachments/assets/5cade474-f9cf-4213-a444-00c85b9d034b" />

---

## View Results:

Prediction Tab: Shows the numerical forecast and the specific "Influence Factors" (e.g., "Adjusted -0.5% due to regulatory risk").

DeepSeek-R1 Logs: Switch to this tab to see the raw "Chain of Thought" and JSON outputs from the AI.

Interactive Plots: Click the buttons to open high-resolution HTML graphs in your browser.

<img width="1341" height="680" alt="image" src="https://github.com/user-attachments/assets/7d6c0855-eb10-4a0e-8df2-9a16e690f782" />

<img width="1318" height="588" alt="Screenshot 2026-01-19 232309" src="https://github.com/user-attachments/assets/7ca84de3-fb4b-4d5e-ad3c-36be712bcbf7" />

<img width="1285" height="556" alt="Screenshot 2026-01-19 232355" src="https://github.com/user-attachments/assets/8947bd9e-10d1-4034-bdc4-1a8e6190ba1a" />

---

## File Structure
lazyapproachfinal.py: The main entry point for the GUI application.

stock_prediction_plot.html: Auto-generated interactive graph of price predictions.

stock_indicators_plot.html: Auto-generated technical analysis dashboard (RSI, MACD).

---

## Troubleshooting
"Ollama Status: Not Running": Ensure you have the Ollama desktop app open or run ollama serve in a separate terminal window.

Freezing GUI: The application is multi-threaded, but heavy LLM inference can slow down older CPUs. Check the terminal console for progress logs.

JSON Parsing Errors: If DeepSeek returns unstructured "thinking" text, the built-in regex fallback will attempt to extract the score. Check the "Logs" tab for details.

License
MIT License - Copyright (c) 2026 Jerome Antony Robin. Free for academic and research use.
