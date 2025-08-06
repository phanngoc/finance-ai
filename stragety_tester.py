"""
MVP Strategy Tester for Vietnamese Stock Market
Uses vnstock for data retrieval and implements common trading strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from vnstock import Vnstock
except ImportError:
    print("Please install vnstock: pip install vnstock")
    exit(1)


class StrategyTester:
    """
    Main class for testing trading strategies on Vietnamese stock market
    """
    
    def __init__(self, initial_capital: float = 100_000_000):  # 100M VND
        """
        Initialize strategy tester
        
        Args:
            initial_capital: Starting capital in VND
        """
        self.initial_capital = initial_capital
        self.stock = Vnstock().stock(symbol="VN30", source="VCI")
        self.data = None
        self.results = {}
        
    def get_stock_data(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data from vnstock
        
        Args:
            symbol: Stock symbol (e.g., 'VCB', 'VIC', 'VHM')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            # Get stock data
            self.stock = Vnstock().stock(symbol=symbol, source="VCI")
            data = self.stock.quote.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
                
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                print(f"Missing required columns in data for {symbol}")
                return pd.DataFrame()
                
            # Clean and prepare data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            print(f"Successfully loaded {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to the data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['close'].rolling(window=5).mean()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Price change and returns
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['close'].diff()
        
        return df
    
    def sma_crossover_strategy(self, data: pd.DataFrame, short_window: int = 10, long_window: int = 20) -> pd.DataFrame:
        """
        Simple Moving Average Crossover Strategy
        
        Args:
            data: DataFrame with price data and indicators
            short_window: Short SMA period
            long_window: Long SMA period
            
        Returns:
            DataFrame with buy/sell signals
        """
        df = data.copy()
        
        # Calculate SMAs if not already present
        df[f'SMA_{short_window}'] = df['close'].rolling(window=short_window).mean()
        df[f'SMA_{long_window}'] = df['close'].rolling(window=long_window).mean()
        
        # Generate signals
        df['signal'] = 0
        df['signal'][short_window:] = np.where(
            df[f'SMA_{short_window}'][short_window:] > df[f'SMA_{long_window}'][short_window:], 1, 0
        )
        df['positions'] = df['signal'].diff()
        
        return df
    
    def rsi_strategy(self, data: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
        """
        RSI Strategy - Buy when oversold, sell when overbought
        
        Args:
            data: DataFrame with RSI indicator
            oversold: RSI level for buy signal
            overbought: RSI level for sell signal
            
        Returns:
            DataFrame with buy/sell signals
        """
        df = data.copy()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['RSI'] < oversold, 'signal'] = 1  # Buy signal
        df.loc[df['RSI'] > overbought, 'signal'] = -1  # Sell signal
        df['positions'] = df['signal'].diff()
        
        return df
    
    def macd_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MACD Strategy - Buy when MACD crosses above signal line
        
        Args:
            data: DataFrame with MACD indicators
            
        Returns:
            DataFrame with buy/sell signals
        """
        df = data.copy()
        
        # Generate signals
        df['signal'] = 0
        df['signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, 0)
        df['positions'] = df['signal'].diff()
        
        return df
    
    def bollinger_bands_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands Strategy - Buy at lower band, sell at upper band
        
        Args:
            data: DataFrame with Bollinger Bands
            
        Returns:
            DataFrame with buy/sell signals
        """
        df = data.copy()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] <= df['BB_lower'], 'signal'] = 1  # Buy signal
        df.loc[df['close'] >= df['BB_upper'], 'signal'] = -1  # Sell signal
        df['positions'] = df['signal'].diff()
        
        return df
    
    def backtest_strategy(self, data: pd.DataFrame, strategy_name: str) -> Dict:
        """
        Backtest a strategy and calculate performance metrics
        
        Args:
            data: DataFrame with signals and positions
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with performance metrics
        """
        df = data.copy()
        
        # Calculate returns
        df['strategy_returns'] = df['signal'].shift(1) * df['price_change']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['buy_hold_returns'] = (1 + df['price_change']).cumprod()
        
        # Calculate portfolio value
        df['portfolio_value'] = self.initial_capital * df['cumulative_returns']
        
        # Performance metrics
        total_return = df['cumulative_returns'].iloc[-1] - 1
        buy_hold_return = df['buy_hold_returns'].iloc[-1] - 1
        
        # Calculate number of trades
        trades = df['positions'].abs().sum() / 2
        
        # Calculate win rate
        winning_trades = df[df['strategy_returns'] > 0]['strategy_returns'].count()
        total_trades = df[df['strategy_returns'] != 0]['strategy_returns'].count()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252) if df['strategy_returns'].std() != 0 else 0
        
        # Maximum drawdown
        rolling_max = df['cumulative_returns'].expanding().max()
        drawdown = (df['cumulative_returns'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        results = {
            'strategy_name': strategy_name,
            'total_return': total_return * 100,
            'buy_hold_return': buy_hold_return * 100,
            'excess_return': (total_return - buy_hold_return) * 100,
            'final_portfolio_value': df['portfolio_value'].iloc[-1],
            'number_of_trades': int(trades),
            'win_rate': win_rate * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'data': df
        }
        
        return results
    
    def run_strategy_comparison(self, symbol: str, start_date: str, end_date: Optional[str] = None) -> Dict:
        """
        Run and compare multiple strategies on a single stock
        
        Args:
            symbol: Stock symbol
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Dictionary with results for all strategies
        """
        print(f"Testing strategies for {symbol}...")
        
        # Get data
        data = self.get_stock_data(symbol, start_date, end_date)
        if data.empty:
            return {}
        
        # Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Test strategies
        strategies = {}
        
        # SMA Crossover Strategy
        try:
            sma_data = self.sma_crossover_strategy(data, 10, 20)
            strategies['SMA_Crossover'] = self.backtest_strategy(sma_data, 'SMA Crossover')
        except Exception as e:
            print(f"Error in SMA strategy: {e}")
        
        # RSI Strategy
        try:
            rsi_data = self.rsi_strategy(data)
            strategies['RSI'] = self.backtest_strategy(rsi_data, 'RSI')
        except Exception as e:
            print(f"Error in RSI strategy: {e}")
        
        # MACD Strategy
        try:
            macd_data = self.macd_strategy(data)
            strategies['MACD'] = self.backtest_strategy(macd_data, 'MACD')
        except Exception as e:
            print(f"Error in MACD strategy: {e}")
        
        # Bollinger Bands Strategy
        try:
            bb_data = self.bollinger_bands_strategy(data)
            strategies['Bollinger_Bands'] = self.backtest_strategy(bb_data, 'Bollinger Bands')
        except Exception as e:
            print(f"Error in Bollinger Bands strategy: {e}")
        
        self.results[symbol] = strategies
        return strategies
    
    def print_results_summary(self, symbol: str):
        """
        Print a summary of strategy performance
        
        Args:
            symbol: Stock symbol to print results for
        """
        if symbol not in self.results:
            print(f"No results found for {symbol}")
            return
        
        print(f"\n{'='*60}")
        print(f"STRATEGY PERFORMANCE SUMMARY FOR {symbol}")
        print(f"{'='*60}")
        print(f"Initial Capital: {self.initial_capital:,.0f} VND")
        print(f"{'='*60}")
        
        strategies = self.results[symbol]
        
        for strategy_name, results in strategies.items():
            print(f"\n{strategy_name.upper()}:")
            print(f"  Total Return: {results['total_return']:.2f}%")
            print(f"  Buy & Hold Return: {results['buy_hold_return']:.2f}%")
            print(f"  Excess Return: {results['excess_return']:.2f}%")
            print(f"  Final Portfolio Value: {results['final_portfolio_value']:,.0f} VND")
            print(f"  Number of Trades: {results['number_of_trades']}")
            print(f"  Win Rate: {results['win_rate']:.2f}%")
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"  {'-'*40}")
    
    def plot_strategy_performance(self, symbol: str, strategy_name: Optional[str] = None):
        """
        Plot strategy performance
        
        Args:
            symbol: Stock symbol
            strategy_name: Specific strategy to plot (if None, plots all)
        """
        if symbol not in self.results:
            print(f"No results found for {symbol}")
            return
        
        strategies = self.results[symbol]
        
        if strategy_name and strategy_name in strategies:
            strategies = {strategy_name: strategies[strategy_name]}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Strategy Performance for {symbol}', fontsize=16)
        
        # Plot 1: Cumulative Returns
        ax1 = axes[0, 0]
        for name, results in strategies.items():
            data = results['data']
            ax1.plot(data.index, data['cumulative_returns'], label=f"{name}")
        
        # Add buy & hold for comparison
        if strategies:
            first_strategy_data = list(strategies.values())[0]['data']
            ax1.plot(first_strategy_data.index, first_strategy_data['buy_hold_returns'], 
                    label='Buy & Hold', linestyle='--', alpha=0.7)
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Portfolio Value
        ax2 = axes[0, 1]
        for name, results in strategies.items():
            data = results['data']
            ax2.plot(data.index, data['portfolio_value'], label=f"{name}")
        
        ax2.set_title('Portfolio Value (VND)')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Performance Metrics Bar Chart
        ax3 = axes[1, 0]
        metrics = ['total_return', 'sharpe_ratio', 'win_rate']
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [strategies[name][metric] for name in strategies.keys()]
            ax3.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(strategies.keys(), rotation=45)
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Max Drawdown
        ax4 = axes[1, 1]
        drawdown_values = [strategies[name]['max_drawdown'] for name in strategies.keys()]
        colors = ['red' if x < 0 else 'green' for x in drawdown_values]
        ax4.bar(strategies.keys(), drawdown_values, color=colors, alpha=0.7)
        ax4.set_title('Maximum Drawdown (%)')
        ax4.set_ylabel('Drawdown %')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to demonstrate the strategy tester
    """
    print("Welcome to Vietnamese Stock Market Strategy Tester!")
    print("This MVP uses vnstock to test common trading strategies.")
    
    # Initialize strategy tester
    tester = StrategyTester(initial_capital=100_000_000)  # 100M VND
    
    # Get user input
    symbol = input("Enter stock symbol (e.g., VCB, VIC, VHM): ").upper()
    
    # Set default dates (1 year of data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Testing strategies for {symbol} from {start_date} to {end_date}")
    
    try:
        # Run strategy comparison
        results = tester.run_strategy_comparison(symbol, start_date, end_date)
        
        if results:
            # Print results
            tester.print_results_summary(symbol)
            
            # Ask if user wants to see plots
            show_plots = input("\nDo you want to see performance plots? (y/n): ").lower()
            if show_plots == 'y':
                tester.plot_strategy_performance(symbol)
        else:
            print(f"No results generated for {symbol}. Please check if the symbol is valid.")
    
    except Exception as e:
        print(f"Error running strategy tests: {e}")


if __name__ == "__main__":
    main()
