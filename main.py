import yfinance as yf
import pandas as pd

# Lấy dữ liệu giá BTC/USD
data = yf.download('BTC-USD', start='2021-01-01', end='2024-01-01')
data = data[['Close']]


from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Tạo chuỗi thời gian đầu vào (lookback = 60)
lookback = 60
X, y = [], []

for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))  # Dự đoán giá

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)

# So sánh thực tế và dự đoán
import matplotlib.pyplot as plt

real_prices = scaler.inverse_transform(y.reshape(-1, 1))

plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.legend()
plt.show()


signals = []

for i in range(1, len(predicted_prices)):
    if predicted_prices[i] > predicted_prices[i - 1]:
        signals.append('Buy')
    else:
        signals.append('Sell')