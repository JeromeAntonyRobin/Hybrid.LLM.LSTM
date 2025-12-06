# Deep Learning Stock Predictor (Prototype Architecture)

## Description
This series documents the engineering of a neural network-based time-series predictor. It progresses from basic LSTM implementations to advanced validation techniques. The focus is on mathematical stability, model architecture selection, and preventing data leakage during testing.

## Version History
* **finalproto2/3:** Implements core LSTM architecture with Dropout layers to prevent overfitting.
* **finalproto4:** Introduces "Average Single Run" logic to mitigate stochastic initialization of neural network weights.
* **finalproto5:** Adds hyperparameter tuning interface (Grid Search) for optimizing layers, units, and learning rates.
* **finalproto6:** Implements "Absolute Testing" (Walk-Forward Validation). Trains on data prior to a specific cutoff date and predicts subsequent data to simulate real-world trading conditions without look-ahead bias.
* **finalproto7:** Adds recursive multi-step forecasting to predict future unknown dates.
* **FinalprotoWui:** A simplified interface version intended for end-user deployment.

## Technical Specifications
* **Framework:** TensorFlow (Keras)
* **Models:** LSTM, Bidirectional LSTM, Conv1D.
* **Preprocessing:** MinMaxScaler (0-1 normalization).
* **Validation:** Out-of-sample testing and recursive prediction.

## Usage
Execute the desired version. Use the "Absolute Testing" tab in later versions for valid performance metrics.
