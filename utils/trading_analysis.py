"""
Trading signal analysis utilities
"""
import numpy as np
import pandas as pd


def generate_trading_signals(predicted_prices):
    """
    Generate buy/sell signals based on predicted prices
    
    Args:
        predicted_prices (numpy.array): Array of predicted prices
        
    Returns:
        list: List of trading signals ('Mua' or 'Bán')
    """
    signals = []
    pred_flat = predicted_prices.flatten()
    
    for i in range(1, len(pred_flat)):
        if pred_flat[i] > pred_flat[i - 1]:
            signals.append('Mua')
        else:
            signals.append('Bán')
    
    return signals


def find_optimal_buy_points(predicted_prices, min_profit_threshold=2.0, lookback=20):
    """
    Find optimal buy points in predicted prices
    
    Args:
        predicted_prices (numpy.array): Array of predicted prices
        min_profit_threshold (float): Minimum profit potential percentage
        lookback (int): Number of points to look ahead for profit calculation
        
    Returns:
        list: List of buy opportunities with details
    """
    pred_flat = predicted_prices.flatten()
    buy_opportunities = []
    today = pd.Timestamp.today()
    
    for i in range(1, min(len(pred_flat) - 1, lookback)):
        if (pred_flat[i] < pred_flat[i-1] and 
            pred_flat[i] < pred_flat[i+1]):
            profit_potential = (np.max(pred_flat[i:]) - pred_flat[i]) / pred_flat[i] * 100
            if profit_potential > min_profit_threshold:
                days_ahead = int(i)
                future_date = today + pd.DateOffset(days=days_ahead)
                
                buy_opportunities.append({
                    'index': i,
                    'price': float(pred_flat[i]),
                    'date': future_date,
                    'profit_potential': profit_potential
                })
    
    # If no opportunities found, use recent minimum
    if not buy_opportunities and len(pred_flat) > 0:
        recent_prices = pred_flat[-10:] if len(pred_flat) >= 10 else pred_flat
        min_recent_idx = np.argmin(recent_prices)
        min_price = recent_prices[min_recent_idx]
        days_ahead = int(min_recent_idx + 1)
        future_date = today + pd.DateOffset(days=days_ahead)
        
        buy_opportunities.append({
            'index': min_recent_idx,
            'price': float(min_price),
            'date': future_date,
            'profit_potential': 15.0  # Default assumption
        })
    
    # Sort by profit potential
    buy_opportunities.sort(key=lambda x: x['profit_potential'], reverse=True)
    return buy_opportunities


def find_optimal_sell_points(predicted_prices, min_risk_threshold=2.0, lookback=20):
    """
    Find optimal sell points in predicted prices
    
    Args:
        predicted_prices (numpy.array): Array of predicted prices
        min_risk_threshold (float): Minimum risk percentage to consider
        lookback (int): Number of points to look ahead for risk calculation
        
    Returns:
        list: List of sell opportunities with details
    """
    pred_flat = predicted_prices.flatten()
    sell_opportunities = []
    today = pd.Timestamp.today()
    
    for i in range(1, min(len(pred_flat) - 1, lookback)):
        if (pred_flat[i] > pred_flat[i-1] and 
            pred_flat[i] > pred_flat[i+1]):
            price_drop = (pred_flat[i] - np.min(pred_flat[i:])) / pred_flat[i] * 100
            if price_drop > min_risk_threshold:
                days_ahead = int(i)
                future_date = today + pd.DateOffset(days=days_ahead)
                
                sell_opportunities.append({
                    'index': i,
                    'price': float(pred_flat[i]),
                    'date': future_date,
                    'risk_level': price_drop
                })
    
    # If no opportunities found, use recent maximum
    if not sell_opportunities and len(pred_flat) > 0:
        recent_prices = pred_flat[-10:] if len(pred_flat) >= 10 else pred_flat
        max_recent_idx = np.argmax(recent_prices)
        max_price = recent_prices[max_recent_idx]
        days_ahead = int(max_recent_idx + 1)
        future_date = today + pd.DateOffset(days=days_ahead)
        
        sell_opportunities.append({
            'index': max_recent_idx,
            'price': float(max_price),
            'date': future_date,
            'risk_level': 10.0  # Default assumption
        })
    
    sell_opportunities.sort(key=lambda x: x['risk_level'], reverse=True)
    return sell_opportunities


def calculate_price_volatility(predicted_prices):
    """
    Calculate price volatility from predicted prices
    
    Args:
        predicted_prices (numpy.array): Array of predicted prices
        
    Returns:
        float: Volatility percentage
    """
    pred_flat = predicted_prices.flatten()
    if len(pred_flat) == 0 or np.mean(pred_flat) == 0:
        return 0.0
    
    return np.std(pred_flat) / np.mean(pred_flat) * 100


def analyze_trend(predicted_prices, window=10):
    """
    Analyze price trend from predicted prices
    
    Args:
        predicted_prices (numpy.array): Array of predicted prices
        window (int): Number of recent points to analyze
        
    Returns:
        dict: Dictionary containing trend direction and strength
    """
    pred_flat = predicted_prices.flatten()
    
    if len(pred_flat) < window:
        recent_trend = pred_flat
    else:
        recent_trend = pred_flat[-window:]
    
    if len(recent_trend) < 2:
        return {
            'direction': 'Không xác định',
            'strength': 0.0
        }
    
    trend_direction = "Tăng" if recent_trend[-1] > recent_trend[0] else "Giảm"
    
    if recent_trend[0] == 0:
        trend_strength = 0.0
    else:
        trend_strength = abs((recent_trend[-1] - recent_trend[0]) / recent_trend[0] * 100)
    
    return {
        'direction': trend_direction,
        'strength': trend_strength
    }


def analyze_trading_signals(predicted_prices, real_prices, dates):
    """
    Comprehensive analysis of trading signals based on predicted prices
    
    Args:
        predicted_prices (numpy.array): Array of predicted prices
        real_prices (numpy.array): Array of real prices
        dates (pandas.DatetimeIndex): Array of dates
        
    Returns:
        dict: Comprehensive trading analysis
    """
    pred_flat = predicted_prices.flatten()
    
    # Calculate volatility
    price_volatility = calculate_price_volatility(predicted_prices)
    
    # Analyze trend
    trend_info = analyze_trend(predicted_prices)
    
    # Find buy and sell opportunities
    buy_opportunities = find_optimal_buy_points(predicted_prices)
    sell_opportunities = find_optimal_sell_points(predicted_prices)
    
    # Find min/max prices and their dates
    min_price_idx = np.argmin(pred_flat)
    max_price_idx = np.argmax(pred_flat)
    
    today = pd.Timestamp.today()
    min_days = int(min_price_idx % 30 + 1)
    max_days = int(max_price_idx % 30 + 1)
    min_price_future_date = today + pd.DateOffset(days=min_days)
    max_price_future_date = today + pd.DateOffset(days=max_days)
    
    trading_analysis = {
        'volatility': price_volatility,
        'trend_direction': trend_info['direction'],
        'trend_strength': trend_info['strength'],
        'best_buy': buy_opportunities[0] if buy_opportunities else None,
        'best_sell': sell_opportunities[0] if sell_opportunities else None,
        'min_price_date': min_price_future_date,
        'max_price_date': max_price_future_date,
        'min_price': float(pred_flat[min_price_idx]),
        'max_price': float(pred_flat[max_price_idx])
    }
    
    return trading_analysis


def calculate_trend_strength_over_time(predicted_prices, window_size=10):
    """
    Calculate trend strength over time using a rolling window
    
    Args:
        predicted_prices (numpy.array): Array of predicted prices
        window_size (int): Size of the rolling window
        
    Returns:
        list: List of trend strength values over time
    """
    pred_flat = predicted_prices.flatten()
    trend_data = []
    
    if len(pred_flat) > window_size:
        for i in range(window_size, len(pred_flat)):
            window_data = pred_flat[i-window_size:i]
            if len(window_data) > 0 and window_data[0] != 0:
                trend_strength = (window_data[-1] - window_data[0]) / window_data[0] * 100
                trend_data.append(trend_strength)
    
    return trend_data


def calculate_signal_distribution(signals):
    """
    Calculate the distribution of buy/sell signals
    
    Args:
        signals (list): List of trading signals
        
    Returns:
        dict: Distribution of signals
    """
    if not signals:
        return {'Mua': 0, 'Bán': 0}
    
    signal_counts = pd.Series(signals).value_counts()
    return {
        'Mua': signal_counts.get('Mua', 0),
        'Bán': signal_counts.get('Bán', 0)
    }


def calculate_potential_profit(buy_info, sell_info):
    """
    Calculate potential profit from buy and sell points
    
    Args:
        buy_info (dict): Buy opportunity information
        sell_info (dict): Sell opportunity information
        
    Returns:
        dict: Profit analysis
    """
    if not buy_info or not sell_info:
        return {
            'profit_percentage': 0.0,
            'profit_per_share': 0.0,
            'is_profitable': False
        }
    
    try:
        buy_price = float(buy_info['price'])
        sell_price = float(sell_info['price'])
        
        if buy_price <= 0:
            return {
                'profit_percentage': 0.0,
                'profit_per_share': 0.0,
                'is_profitable': False
            }
        
        profit_percentage = ((sell_price - buy_price) / buy_price * 100)
        profit_per_share = sell_price - buy_price
        is_profitable = profit_percentage > 0
        
        return {
            'profit_percentage': profit_percentage,
            'profit_per_share': profit_per_share,
            'is_profitable': is_profitable
        }
    except (ValueError, TypeError, ZeroDivisionError):
        return {
            'profit_percentage': 0.0,
            'profit_per_share': 0.0,
            'is_profitable': False
        }
