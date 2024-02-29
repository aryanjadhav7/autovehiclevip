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
def create_data(prices):
     # Calculate daily percent changes and return them
    return np.diff(prices) / prices[:-1] * 100
def main():
    ticker = input("Enter stock ticker: ")
    data = fetch_data(ticker)

    if data.empty:
        print(f"No data found for {ticker}.")
        return

    prices = create_data(data['Close'].values)
    X = prices[:-1].reshape(-1,1)
    y= prices[1:]
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    gp = train_gp(X_scaled, y_scaled)

    # Predict future stock prices for the next 30 days
    prediction = np.zeros(30)
    prediction_prices = np.zeros(30)
    prediction[0]=gp.predict(scaler_X.transform(prices[-1].reshape(-1,1)))
    print(prediction[0])
    prediction_prices[0]=data['Close'].values[-1]+((data['Close'].values[-1]*scaler_y.transform(prediction[0].reshape(-1,1))/100))
    for i in range(1,30):
        curr = prediction[i-1]
        prediction[i]=gp.predict(scaler_X.transform(curr.reshape(-1,1)))
        print(prediction[i])
        prediction_prices[i]=prediction_prices[i-1]+((prediction_prices[i-1]*scaler_y.transform(prediction[i].reshape(-1,1)))/100)

    #X_future = np.arange(len(data), len(data) + 30).reshape(-1, 1)
    #_future_scaled = scaler_y.transform(X_future)

    #y_pred_scaled = gp.predict(X_future_scaled)
    #y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Plotting
    plt.clf()  # Clear the current figure
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'].values, label='Historical Daily Closing Price')
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
    plt.plot(future_dates, prediction_prices, 'r-', label='Predicted Future Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
