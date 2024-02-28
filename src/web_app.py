import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


def forecasting(x, y, scale):
    model: tf.keras.Model = load_model("lstm_model.h5")
    y_pred = model.predict(x=x)
    y_pred = y_pred / scale

    fig = plt.figure(figsize=(12, 6))
    plt.plot(y_pred, "r", label="Predicted")
    plt.plot(y, "b", label="Actual")
    plt.legend()
    return fig, y_pred


def take_testing_data(df, train_rate: float = 0.7):
    testing = df.Close[int(len(df) * train_rate) :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    testing = scaler.fit_transform(np.array(testing).reshape(-1, 1))
    x_test = []
    y_test = []

    for i in range(100, testing.shape[0]):
        x_test.append(testing[i - 100 : i])
        y_test.append(testing[i])

    x_test = np.array(x_test)
    y_test = np.array(y_test) / scaler.scale_[0]

    return x_test, y_test, scaler.scale_[0]


def main():
    plt.style.use("ggplot")
    START = datetime.datetime(2015, 1, 1)
    END = datetime.datetime(2023, 12, 31)

    st.title("Stock Forecasting")

    user_input = st.text_input(
        "Enter Stock Ticker", placeholder="Example: AAPL", value="AAPL"
    )
    ticker = yf.Ticker(user_input)
    history = ticker.history(start=START, end=END)

    if "Dividends" and "Stock Splits" in history:
        df = history.drop(columns=["Dividends", "Stock Splits"])
    else:
        df = history.copy()

    # Describing Data
    st.subheader("Data from 2015-2023")
    st.table(df.describe())

    # Visualizations
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    st.subheader("Closing Price vs Time chart with 100MA and 200MA")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, "b")
    plt.plot(ma100, "r")
    plt.plot(ma200, "g")
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend(["Close", "MA100"])
    plt.title(f"Closing Price of ticker {user_input}")
    st.pyplot(fig)

    st.subheader("Prediction vs Original")
    x_test, y_test, scale = take_testing_data(df)
    fig2, y_pred = forecasting(x=x_test, y=y_test, scale=scale)
    st.pyplot(fig2)


if __name__ == "__main__":
    main()
