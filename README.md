# Air Passengers Forecasting with LSTM

This project implements a time series forecasting model using an LSTM (Long Short-Term Memory) neural network to predict the number of monthly international airline passengers. The project uses the classic "AirPassengers" dataset, which contains monthly total international airline passengers from 1949 to 1960.

## Project Overview

The project is built with Python and PyTorch. It includes the following major steps:

- Data loading and preprocessing
- Time series decomposition (trend, seasonality, residuals)
- Training (or loading) an LSTM forecasting model
- Evaluating predictions with metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score
- Forecasting future values (next 12 months)
- Visualization of actual vs. predicted values and future forecasts

---

## Project Structure

air_passengers_forecasting/
├── data/
│ └── AirPassengers.csv
├── models/
│ └── lstm_air_passengers.pth # Saved PyTorch model
├── notebooks/
├── main.py # Main script to train, test, and forecast
├── utils.py # Helper functions (loading, preprocessing, decomposition)
└── README.md

```

```
