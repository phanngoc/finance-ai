"""
Data loading and processing utilities
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


@st.cache_data
def load_news_data(symbol):
    """
    Load news data for a specific stock symbol
    
    Args:
        symbol (str): Stock symbol (e.g., 'ACB', 'VCB')
        
    Returns:
        DataFrame: News data for the symbol, empty DataFrame if file not found
    """
    try:
        # Construct the file path
        base_path = os.path.dirname(os.path.dirname(__file__))  # Go up two levels from utils
        csv_file_path = os.path.join(base_path, 'data', 'classified_articles', f'{symbol}_articles.csv')
        
        # Check if file exists
        if not os.path.exists(csv_file_path):
            return pd.DataFrame()
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Convert pub_date to datetime if it exists
        if 'pub_date' in df.columns:
            df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
            # Sort by publication date (newest first)
            df = df.sort_values('pub_date', ascending=False)
        
        return df
        
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu tin tức: {str(e)}")
        return pd.DataFrame()


def format_news_for_display(news_df, max_articles=10):
    """
    Format news data for display in Streamlit
    
    Args:
        news_df (DataFrame): News data
        max_articles (int): Maximum number of articles to display
        
    Returns:
        DataFrame: Formatted news data
    """
    if news_df.empty:
        return pd.DataFrame()
    
    # Select and limit articles
    display_df = news_df.head(max_articles).copy()
    
    # Format publication date
    if 'pub_date' in display_df.columns:
        display_df['formatted_date'] = display_df['pub_date'].dt.strftime('%d/%m/%Y %H:%M')
    else:
        display_df['formatted_date'] = 'N/A'
    
    # Create display columns
    formatted_data = []
    for _, row in display_df.iterrows():
        formatted_data.append({
            'Tiêu đề': row.get('title', 'N/A'),
            'Ngày đăng': row.get('formatted_date', 'N/A'),
            'Danh mục': row.get('category', 'N/A'),
            'Chuyên mục': row.get('section', 'N/A'),
            'Độ tin cậy': f"{row.get('confidence_score', 0)}/5" if 'confidence_score' in row else 'N/A',
            'Link': row.get('link', '#')
        })
    
    return pd.DataFrame(formatted_data)
