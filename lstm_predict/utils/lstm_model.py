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
    Prepare data for LSTM model training with multiple features
    
    Args:
        data (DataFrame): Stock data with 'close', 'volume', 'open', 'high', 'low' columns
        lookback (int): Number of previous days to use for prediction
        
    Returns:
        tuple: (X, y, scaler) where X is input features, y is target values, scaler is the fitted scaler
    """
    # Select features for training
    feature_columns = ['close', 'volume', 'open', 'high', 'low']
    available_columns = [col for col in feature_columns if col in data.columns]
    
    if len(available_columns) == 0:
        raise ValueError("No valid feature columns found in data")
    
    # Prepare feature data
    feature_data = data[available_columns].values
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        # Target is still the close price (first column)
        y.append(scaled_data[i, 0])  # close price is always first
    
    X, y = np.array(X), np.array(y)
    # X shape: (samples, lookback, features)
    # y shape: (samples,)
    
    return X, y, scaler


def create_lstm_model(X, y):
    """
    Create and compile LSTM model for stock price prediction with multiple features
    
    Args:
        X (numpy.array): Input features with shape (samples, lookback, features)
        y (numpy.array): Target values
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Keras/TensorFlow is required for LSTM model")
    
    # Get input shape: (lookback, features)
    input_shape = (X.shape[1], X.shape[2])
    
    model = Sequential()
    
    # First LSTM layer with more units for multiple features
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Third LSTM layer
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    
    # Compile model with better optimizer
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
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
    predicted = model.predict(X, verbose=0)
    
    # Ensure predicted has the right shape for inverse_transform
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    elif predicted.shape[1] != 1:
        predicted = predicted.reshape(-1, 1)
    
    # Create dummy array with same shape as original features for inverse_transform
    # The scaler was trained with multiple features, so we need to provide the same shape
    n_features = scaler.n_features_in_
    dummy_array = np.zeros((len(predicted), n_features))
    dummy_array[:, 0] = predicted.flatten()  # Close price is always first feature
    
    # Inverse transform and return only close prices
    predicted_prices = scaler.inverse_transform(dummy_array)
    
    return predicted_prices[:, 0].reshape(-1, 1)  # Return only close prices


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
    differences = real_clean - pred_clean
    absolute_differences = np.abs(differences)
    squared_differences = differences ** 2
    
    mse = float(np.mean(squared_differences))
    mae = float(np.mean(absolute_differences))
    rmse = float(np.sqrt(mse)) if mse > 0 else 0.0
    
    # Calculate MAPE (Mean Absolute Percentage Error) - avoid division by zero
    percentage_errors = []
    for i in range(len(real_clean)):
        if real_clean[i] != 0:
            percentage_errors.append(abs((real_clean[i] - pred_clean[i]) / real_clean[i]) * 100)
    
    if len(percentage_errors) > 0:
        mape = np.mean(percentage_errors)
        accuracy = max(0.0, 100.0 - float(mape))
    else:
        mape = float('inf')
        accuracy = 0.0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'accuracy': accuracy
    }


def predict_future_prices(model, data, scaler, lookback=60, days_ahead=10):
    """
    Predict future prices for the next n days with multiple features
    
    Args:
        model: Trained LSTM model
        data (DataFrame): Historical stock data with multiple features
        scaler: Fitted MinMaxScaler used for training
        lookback (int): Number of previous days to use for prediction
        days_ahead (int): Number of days to predict into the future
        
    Returns:
        numpy.array: Predicted future prices in original scale
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Keras/TensorFlow is required for LSTM model")
    
    # Select the same features used in training
    feature_columns = ['close', 'volume', 'open', 'high', 'low']
    available_columns = [col for col in feature_columns if col in data.columns]
    feature_data = data[available_columns].values
    
    # Get the last 'lookback' days of scaled data
    scaled_data = scaler.transform(feature_data)
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, -1)
    
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    # Predict day by day
    for _ in range(days_ahead):
        # Predict next day
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence: remove first element, add prediction at the end
        # For multiple features, we need to create a new row with predicted close price
        # and use average values for other features
        current_sequence = np.roll(current_sequence, -1, axis=1)
        
        # Create new row with predicted close price and average values for other features
        new_row = np.zeros((1, current_sequence.shape[2]))
        new_row[0, 0] = next_pred[0, 0]  # close price (first feature)
        
        # For other features, use average of recent values or simple trend
        if current_sequence.shape[2] > 1:
            for i in range(1, current_sequence.shape[2]):
                # Use average of recent values for volume and other features
                recent_values = current_sequence[0, -5:, i]  # last 5 days
                new_row[0, i] = np.mean(recent_values)
        
        current_sequence[0, -1] = new_row[0]
    
    # Convert predictions back to original scale
    # Create a dummy array with the same shape as original features
    dummy_array = np.zeros((len(future_predictions), len(available_columns)))
    dummy_array[:, 0] = future_predictions  # close price is first column
    
    future_prices = scaler.inverse_transform(dummy_array)
    
    return future_prices[:, 0]  # Return only close prices
