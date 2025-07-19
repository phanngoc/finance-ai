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
    predicted = model.predict(X, verbose=0)
    
    # Ensure predicted has the right shape for inverse_transform
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)
    elif predicted.shape[1] != 1:
        predicted = predicted.reshape(-1, 1)
    
    print(f"Predicted shape before inverse transform: {predicted.shape}")
    print(f"Sample predicted values (scaled): {predicted[:5].flatten()}")
    
    predicted_prices = scaler.inverse_transform(predicted)
    
    print(f"Predicted prices shape after inverse transform: {predicted_prices.shape}")
    print(f"Sample predicted prices (unscaled): {predicted_prices[:5].flatten()}")
    
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
    
    # Debug prints
    print(f"Real prices shape: {real_flat.shape}, range: {real_flat.min():.2f} - {real_flat.max():.2f}")
    print(f"Predicted prices shape: {pred_flat.shape}, range: {pred_flat.min():.2f} - {pred_flat.max():.2f}")
    print(f"Sample real values: {real_flat[:5]}")
    print(f"Sample predicted values: {pred_flat[:5]}")
    
    # Remove NaN and inf values
    valid_indices = ~(np.isnan(real_flat) | np.isnan(pred_flat) | 
                     np.isinf(real_flat) | np.isinf(pred_flat))
    real_clean = real_flat[valid_indices]
    pred_clean = pred_flat[valid_indices]
    
    print(f"Valid data points: {len(real_clean)} out of {len(real_flat)}")
    
    if len(real_clean) == 0:
        return {
            'mse': float('inf'),
            'rmse': float('inf'),
            'mae': float('inf'),
            'mape': float('inf'),
            'accuracy': 0.0
        }
    
    # Calculate metrics step by step with detailed debugging
    differences = real_clean - pred_clean
    absolute_differences = np.abs(differences)
    squared_differences = differences ** 2
    
    print(f"Sample differences: {differences[:10]}")
    print(f"Sample absolute differences: {absolute_differences[:10]}")
    print(f"Sample squared differences: {squared_differences[:10]}")
    print(f"Min/Max differences: {differences.min():.4f} / {differences.max():.4f}")
    print(f"Min/Max absolute differences: {absolute_differences.min():.4f} / {absolute_differences.max():.4f}")
    
    # Check if all differences are zero
    if np.allclose(differences, 0, atol=1e-8):
        print("WARNING: All differences are essentially zero!")
        print("This suggests predicted and real values are identical")
    
    mse = float(np.mean(squared_differences))
    mae = float(np.mean(absolute_differences))
    rmse = float(np.sqrt(mse)) if mse > 0 else 0.0
    
    print(f"Final MSE: {mse}")
    print(f"Final MAE: {mae}")
    print(f"Final RMSE: {rmse}")
    
    # Additional check - manually verify calculation
    manual_mae = 0.0
    manual_mse = 0.0
    for i in range(min(len(real_clean), len(pred_clean))):
        diff = abs(real_clean[i] - pred_clean[i])
        manual_mae += diff
        manual_mse += diff ** 2
    
    if len(real_clean) > 0:
        manual_mae /= len(real_clean)
        manual_mse /= len(real_clean)
        
    print(f"Manual verification - MAE: {manual_mae}, MSE: {manual_mse}")
    
    # Use manual calculation if numpy result is suspicious
    if mae == 0 and manual_mae > 0:
        mae = manual_mae
        mse = manual_mse
        rmse = np.sqrt(mse)
        print(f"Using manual calculation - MAE: {mae}, RMSE: {rmse}")
    
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
    Predict future prices for the next n days
    
    Args:
        model: Trained LSTM model
        data (DataFrame): Historical stock data with 'close' column
        scaler: Fitted MinMaxScaler used for training
        lookback (int): Number of previous days to use for prediction
        days_ahead (int): Number of days to predict into the future
        
    Returns:
        numpy.array: Predicted future prices in original scale
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Keras/TensorFlow is required for LSTM model")
    
    # Get the last 'lookback' days of scaled data
    scaled_data = scaler.transform(data[['close']].values)
    last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    # Predict day by day
    for _ in range(days_ahead):
        # Predict next day
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence: remove first element, add prediction at the end
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
    
    # Convert predictions back to original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_predictions)
    
    return future_prices.flatten()
