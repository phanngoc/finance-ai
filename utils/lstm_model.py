"""
LSTM Model utilities for stock price prediction
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


@st.cache_data
def prepare_lstm_data(data, lookback=60):
    """
    Prepare data for LSTM model training
    
    Args:
        data (DataFrame): Stock data with 'close' column
        lookback (int): Number of previous days to use for prediction
        
    Returns:
        tuple: (X, y, scaler) where X is input features, y is target values, scaler is the fitted scaler
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['close']].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


def create_lstm_model(X, y):
    """
    Create and compile LSTM model for stock price prediction
    
    Args:
        X (numpy.array): Input features
        y (numpy.array): Target values
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Keras/TensorFlow is required for LSTM model")
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(model, X, y, epochs=20, batch_size=32, verbose=0):
    """
    Train the LSTM model
    
    Args:
        model: Compiled LSTM model
        X (numpy.array): Input features
        y (numpy.array): Target values
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        verbose (int): Verbosity mode
        
    Returns:
        keras.Model: Trained model
    """
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def make_predictions(model, X, scaler):
    """
    Make predictions using trained LSTM model
    
    Args:
        model: Trained LSTM model
        X (numpy.array): Input features
        scaler: Fitted MinMaxScaler
        
    Returns:
        numpy.array: Predicted prices in original scale
    """
    predicted = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted)
    return predicted_prices


def calculate_model_accuracy(real_prices, predicted_prices):
    """
    Calculate various accuracy metrics for the model
    
    Args:
        real_prices (numpy.array): Actual prices
        predicted_prices (numpy.array): Predicted prices
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Flatten arrays to ensure 1D
    real_flat = real_prices.flatten()
    pred_flat = predicted_prices.flatten()
    
    # Remove NaN and inf values
    valid_indices = ~(np.isnan(real_flat) | np.isnan(pred_flat) | 
                     np.isinf(real_flat) | np.isinf(pred_flat))
    real_clean = real_flat[valid_indices]
    pred_clean = pred_flat[valid_indices]
    
    if len(real_clean) == 0:
        return {
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'mape': float('inf'),
            'accuracy': 0.0
        }
    
    # Calculate metrics
    mse = np.mean((real_clean - pred_clean) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(real_clean - pred_clean))
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((real_clean - pred_clean) / real_clean)) * 100
    accuracy = max(0.0, 100.0 - float(mape))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'accuracy': accuracy
    }
