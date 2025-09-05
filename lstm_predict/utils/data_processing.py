"""
Data loading and processing utilities for LSTM Stock Prediction
"""
import pandas as pd
import streamlit as st
from vnstock import Vnstock
import os
from datetime import datetime


@st.cache_data
def load_stock_data(symbol, start_date, end_date):
    """
    Load stock data from VCI source
    
    Args:
        symbol (str): Stock symbol
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        tuple: (DataFrame, error_message) where DataFrame is stock data and error_message is None if successful
    """
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df = stock.quote.history(
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'), 
            interval='1D'
        )
        return df, None
    except Exception as e:
        return None, str(e)


def calculate_price_change(df):
    """
    Calculate price change and percentage change
    
    Args:
        df (DataFrame): Stock data with 'close' column
        
    Returns:
        tuple: (price_change, change_percent)
    """
    if len(df) <= 1:
        return 0, 0
    
    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
    change_percent = (price_change / df['close'].iloc[-2] * 100)
    
    return price_change, change_percent


def get_basic_stats(df):
    """
    Get basic statistics for stock data
    
    Args:
        df (DataFrame): Stock data
        
    Returns:
        dict: Dictionary containing basic statistics
    """
    latest_price = df['close'].iloc[-1]
    avg_volume = df['volume'].mean()
    price_change, change_percent = calculate_price_change(df)
    
    return {
        'total_days': len(df),
        'latest_price': latest_price,
        'price_change': price_change,
        'change_percent': change_percent,
        'avg_volume': avg_volume
    }


def prepare_prediction_dataframe(prediction_dates, real_prices, predicted_prices):
    """
    Prepare DataFrame for prediction comparison
    
    Args:
        prediction_dates: Array of dates
        real_prices: Array of real prices
        predicted_prices: Array of predicted prices
        
    Returns:
        DataFrame: Comparison DataFrame
    """
    prediction_df = pd.DataFrame({
        'date': prediction_dates,
        'actual': real_prices.flatten(),
        'predicted': predicted_prices.flatten()
    })
    
    return prediction_df


def format_prediction_table(prediction_df, num_rows=10):
    """
    Format prediction DataFrame for display
    
    Args:
        prediction_df (DataFrame): Prediction data
        num_rows (int): Number of rows to show
        
    Returns:
        DataFrame: Formatted DataFrame
    """
    recent_predictions = prediction_df.tail(num_rows).copy()
    recent_predictions['difference'] = recent_predictions['predicted'] - recent_predictions['actual']
    
    # Calculate accuracy with zero division handling
    def calculate_accuracy(actual, predicted):
        if actual == 0:
            return 0
        return max(0, (1 - abs(predicted - actual) / abs(actual)) * 100)
    
    recent_predictions['accuracy'] = recent_predictions.apply(
        lambda row: calculate_accuracy(row['actual'], row['predicted']), axis=1
    )
    
    # Store numeric values for backup
    actual_backup = recent_predictions['actual'].copy()
    predicted_backup = recent_predictions['predicted'].copy()
    difference_backup = recent_predictions['difference'].copy()
    accuracy_backup = recent_predictions['accuracy'].copy()
    
    # Format for display
    recent_predictions['actual'] = actual_backup.apply(lambda x: f"{x:,.0f}")
    recent_predictions['predicted'] = predicted_backup.apply(lambda x: f"{x:,.0f}")
    recent_predictions['difference'] = difference_backup.apply(lambda x: f"{x:+,.0f}")
    recent_predictions['accuracy'] = accuracy_backup.apply(lambda x: f"{x:.1f}%")
    
    recent_predictions.columns = ['Ngày', 'Giá thực tế (VND)', 'Giá dự đoán (VND)', 'Chênh lệch (VND)', 'Độ chính xác']
    
    return recent_predictions
