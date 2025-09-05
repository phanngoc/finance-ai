"""
Plotting utilities for financial charts in LSTM Stock Prediction
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_combined_chart(df, symbol):
    """
    Create combined candlestick and volume chart
    
    Args:
        df (DataFrame): Stock data
        symbol (str): Stock symbol
        
    Returns:
        plotly.Figure: Combined chart
    """
    fig_combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'Giá cổ phiếu {symbol}', 'Khối lượng giao dịch'),
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig_combined.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig_combined.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig_combined.update_layout(
        title=f'Phân tích cổ phiếu {symbol}',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=False
    )
    
    fig_combined.update_xaxes(title_text="Ngày", row=2, col=1)
    fig_combined.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig_combined.update_yaxes(title_text="Khối lượng", row=2, col=1)
    
    return fig_combined


def create_comparison_chart(prediction_df):
    """
    Create comparison chart between actual and predicted prices
    
    Args:
        prediction_df (DataFrame): Prediction comparison data
        
    Returns:
        plotly.Figure: Comparison chart
    """
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatter(
        x=prediction_df['date'],
        y=prediction_df['actual'],
        mode='lines',
        name='Giá thực tế',
        line=dict(color='blue')
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=prediction_df['date'],
        y=prediction_df['predicted'],
        mode='lines',
        name='Giá dự đoán',
        line=dict(color='red', dash='dash')
    ))
    
    fig_comparison.update_layout(
        title='So sánh Giá thực tế vs Dự đoán',
        xaxis_title='Ngày',
        yaxis_title='Giá (VND)',
        height=400
    )
    
    return fig_comparison


def create_trading_signals_chart(prediction_df, signals, trading_info):
    """
    Create trading signals chart with buy/sell points
    
    Args:
        prediction_df (DataFrame): Prediction data
        signals (list): Trading signals
        trading_info (dict): Trading analysis information
        
    Returns:
        plotly.Figure: Trading signals chart
    """
    fig_trading = go.Figure()
    
    # Add predicted price line
    fig_trading.add_trace(go.Scatter(
        x=prediction_df['date'],
        y=prediction_df['predicted'],
        mode='lines',
        name='Giá dự đoán',
        line=dict(color='blue', width=2)
    ))
    
    # Add buy and sell signals
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    
    # Ensure signals and dates alignment
    min_length = min(len(signals), len(prediction_df['date']) - 1)
    
    for i in range(min_length):
        signal_idx = i + 1  # signals start from day 2
        if signal_idx < len(prediction_df):
            if signals[i] == 'Mua':
                buy_dates.append(prediction_df['date'].iloc[signal_idx])
                buy_prices.append(prediction_df['predicted'].iloc[signal_idx])
            else:
                sell_dates.append(prediction_df['date'].iloc[signal_idx])
                sell_prices.append(prediction_df['predicted'].iloc[signal_idx])
    
    # Add buy signals
    if buy_dates:
        fig_trading.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='Tín hiệu MUA',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            showlegend=True
        ))
    
    # Add sell signals
    if sell_dates:
        fig_trading.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name='Tín hiệu BÁN',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            showlegend=True
        ))
    
    # Add optimal buy point if available
    if trading_info.get('best_buy') and trading_info['best_buy'].get('date'):
        try:
            buy_date = trading_info['best_buy']['date']
            buy_price = float(trading_info['best_buy']['price'])
            
            today = pd.Timestamp.today()
            one_year_ago = today - pd.DateOffset(days=365)
            if buy_date >= one_year_ago:
                fig_trading.add_trace(go.Scatter(
                    x=[buy_date],
                    y=[buy_price],
                    mode='markers',
                    name='Điểm MUA tối ưu',
                    marker=dict(color='darkgreen', size=15, symbol='star'),
                    showlegend=True
                ))
        except Exception:
            pass  # Skip if can't display optimal buy point
    
    # Add optimal sell point if available
    if trading_info.get('best_sell') and trading_info['best_sell'].get('date'):
        try:
            sell_date = trading_info['best_sell']['date']
            sell_price = float(trading_info['best_sell']['price'])
            
            today = pd.Timestamp.today()
            one_year_ago = today - pd.DateOffset(days=365)
            if sell_date >= one_year_ago:
                fig_trading.add_trace(go.Scatter(
                    x=[sell_date],
                    y=[sell_price],
                    mode='markers',
                    name='Điểm BÁN tối ưu',
                    marker=dict(color='darkred', size=15, symbol='star'),
                    showlegend=True
                ))
        except Exception:
            pass  # Skip if can't display optimal sell point
    
    fig_trading.update_layout(
        title='Tín hiệu Mua/Bán trên Biểu đồ Giá',
        xaxis_title='Ngày',
        yaxis_title='Giá (VND)',
        height=500,
        showlegend=True
    )
    
    return fig_trading


def create_signals_pie_chart(signals):
    """
    Create pie chart for signal distribution
    
    Args:
        signals (list): Trading signals
        
    Returns:
        plotly.Figure: Pie chart
    """
    if not signals:
        return None
    
    signal_counts = pd.Series(signals).value_counts()
    fig_signals = px.pie(
        values=signal_counts.values,
        names=signal_counts.index,
        title="Phân bổ tín hiệu Mua/Bán",
        color_discrete_map={'Mua': 'green', 'Bán': 'red'}
    )
    
    return fig_signals


def create_trend_strength_chart(trend_data, prediction_dates, window_size=10):
    """
    Create trend strength chart over time
    
    Args:
        trend_data (list): Trend strength values
        prediction_dates: Date index
        window_size (int): Window size used for calculation
        
    Returns:
        plotly.Figure: Trend strength chart
    """
    if not trend_data:
        return None
    
    available_dates = len(prediction_dates) - window_size
    if available_dates <= 0:
        return None
    
    trend_dates = prediction_dates[window_size:window_size + len(trend_data)]
    
    fig_trend = px.bar(
        x=trend_dates,
        y=trend_data,
        title="Cường độ Xu hướng (%)",
        color=trend_data,
        color_continuous_scale=['red', 'yellow', 'green']
    )
    fig_trend.update_layout(height=400)
    
    return fig_trend


def create_accuracy_scatter_plot(real_prices, predicted_prices):
    """
    Create scatter plot for model accuracy visualization
    
    Args:
        real_prices (numpy.array): Actual prices
        predicted_prices (numpy.array): Predicted prices
        
    Returns:
        plotly.Figure: Accuracy scatter plot
    """
    real_flat = real_prices.flatten()
    pred_flat = predicted_prices.flatten()
    
    # Remove invalid values
    valid_indices = ~(np.isnan(real_flat) | np.isnan(pred_flat) | 
                     np.isinf(real_flat) | np.isinf(pred_flat))
    real_clean = real_flat[valid_indices]
    pred_clean = pred_flat[valid_indices]
    
    if len(real_clean) == 0:
        return None
    
    fig_accuracy = px.scatter(
        x=real_clean, 
        y=pred_clean,
        title="Tương quan Giá thực tế vs Dự đoán",
        labels={'x': 'Giá thực tế (VND)', 'y': 'Giá dự đoán (VND)'}
    )
    
    # Add perfect prediction line
    min_val = min(np.min(real_clean), np.min(pred_clean))
    max_val = max(np.max(real_clean), np.max(pred_clean))
    fig_accuracy.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Dự đoán hoàn hảo',
        line=dict(color='red', dash='dash')
    ))
    
    return fig_accuracy


def create_future_prediction_chart(historical_data, future_predictions, future_dates, symbol):
    """
    Create a chart showing historical prices and future predictions
    
    Args:
        historical_data (DataFrame): Historical stock data
        future_predictions (numpy.array): Predicted future prices
        future_dates (list): Dates for future predictions
        symbol (str): Stock symbol
        
    Returns:
        plotly.Figure: Chart with historical and future prices
    """
    fig = go.Figure()
    
    # Add historical close prices
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['close'],
        mode='lines',
        name='Giá thực tế',
        line=dict(color='blue', width=2)
    ))
    
    # Add future predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines+markers',
        name='Dự đoán 10 ngày tới',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=6, color='red')
    ))
    
    # Add a vertical line to separate historical and predicted data
    last_historical_date = historical_data.index[-1]
    fig.add_vline(
        x=last_historical_date, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Ngày hiện tại"
    )
    
    # Add connecting line between last historical point and first prediction
    fig.add_trace(go.Scatter(
        x=[last_historical_date, future_dates[0]],
        y=[historical_data['close'].iloc[-1], future_predictions[0]],
        mode='lines',
        name='Kết nối',
        line=dict(color='orange', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'📋 Dự đoán 10 ngày gần nhất cho {symbol}',
        xaxis_title='Ngày',
        yaxis_title='Giá đóng cửa (VND)',
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig
