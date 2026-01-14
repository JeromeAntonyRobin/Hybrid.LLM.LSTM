# Human-Centric Hybrid Prediction Engine (LLM-LSTM)


## *Project Goal*

To develop a hybrid context-aware prediction logic that bridges the gap between rigid mathematical forecasting and human behavioral reality.

The core objective is to create a system that doesn't just process numbers, but "understands" the environment by taking emotions, irrationality, and human quirks into consideration. Unlike standard statistical models, this engine quantifies abstract human variables such as fear, greed, uncertainty, and sentiment and integrates them directly into a Deep Learning time-series model (LSTM).

While this repository currently demonstrates this logic via Stock Market Prediction (where human emotion drives volatility), the underlying architecture is deployment-agnostic and designed to predict any trend influenced by both historical data and human sentiment.

?? System Concept: The "Ghost in the Machine"
The logic operates on the premise that future events are driven by two forces:

The Trend (Mathematical): What happened yesterday statistically influences what happens tomorrow.

The Context (Emotional): How humans feel about today influences what they do tomorrow.

This system creates a "Digital Cortex" (LLM) that applies an emotional bias to a "Digital Cerebellum" (LSTM) to generate a prediction that is mathematically sound but psychologically adjusted.



*Architecture Overview*

1. The Quantitative Core (The Logical Brain)
Component: Multivariate LSTM (Long Short-Term Memory).

Role: Analyzes hard data (Time-Series, Volume, Historical Patterns).

Function: It creates a baseline prediction based on pure logic and historical precedent.

"Human" Trait: Memory & Pattern Recognition.

2. The Qualitative Layer (The Emotional Brain)
Component: Local Large Language Models (DeepSeek-R1 / Llama 3).

Role: Analyzes soft data (News, Social Sentiment, Global Context).

Function: It reads unstructured text to determine the "mood" of the environment.

Sentiment Analysis: Is the world happy or anxious?

Risk Perception: Are there looming threats (lawsuits, wars, regulations)?

Contextual Relevance: Does this news actually matter to us?

"Human" Trait: Emotion, Intuition, & Fear Assessment.

3. The Synthesis Logic (The Decision)
Component: Weighted Adjustment Algorithm.

Role: Merges Logic and Emotion.

Mechanism: Final Prediction = Logical Baseline * (1 + Emotional Bias + Risk Penalty)

Outcome: A forecast that reflects not just what should happen according to the numbers, but what likely will happen given the current human state of mind.



*Implementation Case Study: Financial Markets*

This repository creates a proof-of-concept using Financial Markets, as they are the perfect sandbox for testing "Human + Math" dynamics.

Input: OHLCV Market Data + Financial News Feeds.

Processing:

LSTM calculates price momentum.

LLM calculates "Market Fear/Greed" and "Macro-Economic Optimism."

Output: A future price curve with confidence intervals that expands or contracts based on the level of detected "Uncertainty."

PS:
I will be working on this Hybrid LLM LSTM project more in future, the only thing that is stopping me from developing it nnow is computational power, Until then See ya!!!
