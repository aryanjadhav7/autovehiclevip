import yfinance as yf
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def fetch_data(ticker, start="2020-01-01", end=None):
    """Fetch historical stock data using the yfinance API."""
    if end is None:
        end = pd.to_datetime("today").strftime("%Y-%m-%d")  # Get the most recent data
    data = yf.download(ticker, start=start, end=end)
    return data

def train_gp(X, y):
    """Train a Gaussian Process Regressor."""
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    return gp

def main():
    ticker = input("Enter stock ticker: ")
    data = fetch_data(ticker)

    if data.empty:
        print(f"No data found for {ticker}.")
        return

    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    gp = train_gp(X_scaled, y_scaled)

    # Predict future stock prices for the next 30 days
    X_future = np.arange(len(data), len(data) + 30).reshape(-1, 1)
    X_future_scaled = scaler_X.transform(X_future)

    y_pred_scaled = gp.predict(X_future_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Plotting
    plt.clf()  # Clear the current figure
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, y, label='Historical Daily Closing Price')
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
    plt.plot(future_dates, y_pred, 'r-', label='Predicted Future Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
