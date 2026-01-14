<h1 align="center"><b>Hybrid LLM-LSTM: The Human-Centric Engine</b></h1>
<p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
    <a href="https://github.com/JeromeAntonyRobin/Hybrid.LLM.LSTM"><img src="https://img.shields.io/badge/Status-Experimental-orange.svg" alt="Status: Experimental"/></a>
</p>

A hybrid prediction engine that integrates quantitative Deep Learning (LSTM) with qualitative Large Language Models (LLM). The system generates a mathematical baseline forecast using time-series data and dynamically adjusts it based on quantified human behavioral factors, such as sentiment, risk severity, and macroeconomic outlook.

# Dual DLL Architecture
The prediction engine has a LLM and a LSTM to process data:
1. **The Trend (Mathematical):** Historical data patterns (The "Digital Cerebellum").
2. **The Context (Emotional):** Human behavior and environmental sentiment (The "Digital Cortex").

This engine creates a synthesis where mathematical sound logic is adjusted by psychological reality.

---
# Architecture Overview
### 1. The Quantitative Core (Logical Brain)
- **Component:** Multivariate LSTM.
- **Function:** Analyzes hard time-series data to create a statistical baseline.

### 2. The Qualitative Layer (Emotional Brain)
- **Component:** Local LLM (DeepSeek-R1 / Llama 3).
- **Function:** Reads unstructured news/sentiment to determine the environmental "mood" and risk factors.

### 3. The Synthesis Logic (The Decision)
- **Component:** Weighted Adjustment Algorithm.
- **Logic:** `Final Prediction = Baseline * (1 + Emotional Bias + Risk Penalty)`

## **Outcome:** 

A forecast that reflects not just what should happen according to the numbers, but what likely will happen given the current human state of mind.



---

## *Implementation Case Study: Financial Markets*

This repository creates a proof-of-concept using Financial Markets, as they are the perfect sandbox for testing "Human + Math" dynamics and have decades worth of OHLCV data.

**Input:** OHLCV Market Data + Financial News Feeds.


* **Processing:**
* LSTM calculates price momentum.


* LLM calculates "Market Fear/Greed" and "Macro-Economic Optimism".


**Output:** A future price curve with confidence intervals that expands or contracts based on the level of detected "Uncertainty".


# Still In Progress
Project development is currently paced by **hardware limitations**, as running local LLMs alongside deep learning models is highly resource-intensive.

## [Click Here](./hybml/hybridseries/finalthatworks/README.md) for the latest stable prototype engine

## License  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
MIT License - Use it. Fork it. Modify it. Built for behavioral and statistical research.
