"""
Configuration file for Strategy Tester
Contains default parameters and settings that can be easily modified
"""

# Default Capital Settings
DEFAULT_INITIAL_CAPITAL = 100_000_000  # 100M VND

# Strategy Parameters
STRATEGY_PARAMS = {
    'sma_crossover': {
        'short_window': 10,
        'long_window': 20
    },
    'rsi': {
        'period': 14,
        'oversold': 30,
        'overbought': 70
    },
    'macd': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    'bollinger_bands': {
        'period': 20,
        'std_dev': 2
    }
}

# Technical Indicators Settings
INDICATORS_CONFIG = {
    'sma_periods': [5, 10, 20, 50, 200],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'bollinger_period': 20,
    'bollinger_std': 2
}

# Backtesting Settings
BACKTEST_CONFIG = {
    'transaction_cost': 0.0015,  # 0.15% per trade (optional to include)
    'slippage': 0.001,  # 0.1% slippage (optional to include)
    'min_trade_value': 1_000_000,  # Minimum 1M VND per trade
    'position_sizing': 'full_capital'  # 'full_capital' or 'fixed_amount'
}

# Data Source Settings
DATA_CONFIG = {
    'source': 'VCI',  # vnstock data source
    'default_period_days': 365,  # Default 1 year of data
    'min_data_points': 50,  # Minimum data points required
    'required_columns': ['open', 'high', 'low', 'close', 'volume']
}

# Popular Vietnamese Stocks for Testing
POPULAR_STOCKS = {
    'banking': ['VCB', 'CTG', 'TCB', 'ACB', 'MBB', 'STB', 'VPB'],
    'real_estate': ['VIC', 'VHM', 'NVL', 'VRE', 'PDR', 'DXG', 'IJC'],
    'technology': ['FPT', 'CMG', 'ELC', 'ITD', 'SAM', 'VGI'],
    'energy': ['GAS', 'POW', 'PVD', 'PVS', 'PVG', 'BSR'],
    'materials': ['HPG', 'HSG', 'NKG', 'HRC', 'TVS', 'VCA'],
    'consumer_goods': ['SAB', 'MWG', 'PNJ', 'MSN', 'VNM', 'KDC'],
    'industrials': ['GMD', 'HAG', 'HCM', 'SCS', 'TNA', 'VPI'],
    'utilities': ['REE', 'PC1', 'HND', 'SBA', 'IDC', 'VCG']
}

# VN30 Index Components (most liquid stocks)
VN30_STOCKS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'CTR', 'FPT', 'GAS', 'GVR', 'HDB',
    'HPG', 'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SSI', 'STB', 'TCB',
    'TPB', 'VCB', 'VHM', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE', 'VGC', 'PDR'
]

# Performance Metrics Configuration
METRICS_CONFIG = {
    'risk_free_rate': 0.0,  # Assume 0% risk-free rate for Sharpe calculation
    'trading_days_per_year': 252,
    'display_decimals': 2,
    'percentage_format': True
}

# Plotting Configuration
PLOT_CONFIG = {
    'figure_size': (15, 10),
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'style': 'seaborn-v0_8',
    'show_grid': True,
    'save_plots': False,
    'plot_directory': 'data/screenshots/'
}

# Default Date Ranges for Testing
DATE_RANGES = {
    'short_term': 90,    # 3 months
    'medium_term': 180,  # 6 months
    'long_term': 365,    # 1 year
    'extended': 730      # 2 years
}
