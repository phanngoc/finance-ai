# Vietnamese Stock Market Strategy Tester MVP

A comprehensive tool for backtesting trading strategies on Vietnamese stocks using the `vnstock` package.

## 🚀 Features

### Core Functionality
- **Multiple Trading Strategies**: SMA Crossover, RSI, MACD, Bollinger Bands
- **Comprehensive Backtesting**: Full performance analysis with multiple metrics
- **Vietnamese Market Data**: Uses `vnstock` for reliable local market data
- **Performance Visualization**: Interactive charts and plots
- **Extensible Architecture**: Easy to add custom strategies

### Supported Strategies

1. **SMA Crossover Strategy**
   - Buy when short SMA (10-day) crosses above long SMA (20-day)
   - Sell when short SMA crosses below long SMA

2. **RSI Strategy**
   - Buy when RSI < 30 (oversold)
   - Sell when RSI > 70 (overbought)

3. **MACD Strategy**
   - Buy when MACD line crosses above signal line
   - Sell when MACD line crosses below signal line

4. **Bollinger Bands Strategy**
   - Buy when price touches lower band
   - Sell when price touches upper band

### Performance Metrics

- **Total Return**: Overall strategy performance
- **Excess Return**: Strategy return vs buy-and-hold
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest portfolio decline
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Total trading frequency

## 📦 Installation

1. **Install Required Packages**:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `vnstock`: Vietnamese stock market data
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Additional analysis tools

2. **Verify vnstock Installation**:
```python
from vnstock import Vnstock
stock = Vnstock().stock(symbol="VCB", source="VCI")
print("vnstock installed successfully!")
```

## 🎯 Quick Start

### Basic Usage

```python
from stragety_tester import StrategyTester
from datetime import datetime, timedelta

# Initialize with 100M VND capital
tester = StrategyTester(initial_capital=100_000_000)

# Test strategies on VCB (Vietcombank)
symbol = "VCB"
start_date = "2023-01-01"
end_date = "2024-01-01"

# Run all strategies
results = tester.run_strategy_comparison(symbol, start_date, end_date)

# Print performance summary
tester.print_results_summary(symbol)

# Show performance plots
tester.plot_strategy_performance(symbol)
```

### Testing Multiple Stocks

```python
stocks = ["VCB", "VIC", "VHM", "FPT", "GAS"]

for symbol in stocks:
    print(f"\\nTesting {symbol}...")
    results = tester.run_strategy_comparison(symbol, start_date, end_date)
    tester.print_results_summary(symbol)
```

### Custom Strategy Example

```python
class CustomTester(StrategyTester):
    def golden_cross_strategy(self, data):
        """Golden Cross: SMA50 crosses above SMA200"""
        df = data.copy()
        
        # Calculate SMAs
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['SMA_200'] = df['close'].rolling(200).mean()
        
        # Generate signals
        df['signal'] = 0
        df['signal'][200:] = np.where(
            df['SMA_50'][200:] > df['SMA_200'][200:], 1, 0
        )
        df['positions'] = df['signal'].diff()
        
        return df

# Use custom strategy
custom_tester = CustomTester()
data = custom_tester.get_stock_data("VIC", "2023-01-01", "2024-01-01")
data = custom_tester.add_technical_indicators(data)
strategy_data = custom_tester.golden_cross_strategy(data)
results = custom_tester.backtest_strategy(strategy_data, "Golden Cross")
```

## 📊 Popular Vietnamese Stocks

The tool works with any Vietnamese stock symbol. Here are some popular choices:

### Banking Sector
- **VCB**: Vietcombank
- **CTG**: VietinBank  
- **TCB**: Techcombank
- **ACB**: Asia Commercial Bank

### Real Estate & Construction
- **VIC**: Vingroup
- **VHM**: Vinhomes
- **NVL**: Novaland
- **VRE**: Vincom Retail

### Technology & Telecom
- **FPT**: FPT Corporation
- **CMG**: CMC Global
- **ELC**: ELCOM

### Energy & Materials
- **GAS**: Gas Petrolimex
- **HPG**: Hoa Phat Group
- **POW**: PetroVietnam Power

### Consumer Goods
- **SAB**: Sabeco
- **MWG**: Mobile World
- **PNJ**: Phu Nhuan Jewelry

## 🔧 Advanced Usage

### Running the Demo

```bash
python demo_strategy_tester.py
```

Choose from:
1. Single stock strategy test
2. Multiple stocks comparison
3. Custom strategy example
4. All demos

### Command Line Interface

```bash
python stragety-tester.py
```

Follow the interactive prompts to:
- Enter stock symbol
- View results summary
- Generate performance plots

### Customizing Parameters

```python
# Adjust strategy parameters
sma_data = tester.sma_crossover_strategy(data, short_window=5, long_window=15)
rsi_data = tester.rsi_strategy(data, oversold=25, overbought=75)

# Modify initial capital
tester = StrategyTester(initial_capital=50_000_000)  # 50M VND

# Use different date ranges
results = tester.run_strategy_comparison("VCB", "2022-01-01", "2023-12-31")
```

## 📈 Example Output

```
============================================================
STRATEGY PERFORMANCE SUMMARY FOR VCB
============================================================
Initial Capital: 100,000,000 VND
============================================================

SMA CROSSOVER:
  Total Return: 15.67%
  Buy & Hold Return: 12.34%
  Excess Return: 3.33%
  Final Portfolio Value: 115,670,000 VND
  Number of Trades: 23
  Win Rate: 65.22%
  Sharpe Ratio: 1.42
  Max Drawdown: -8.45%
  ----------------------------------------

RSI:
  Total Return: 18.92%
  Buy & Hold Return: 12.34%
  Excess Return: 6.58%
  Final Portfolio Value: 118,920,000 VND
  Number of Trades: 31
  Win Rate: 58.06%
  Sharpe Ratio: 1.67
  Max Drawdown: -12.33%
  ----------------------------------------
```

## ⚠️ Important Notes

### Data Limitations
- **Market Hours**: vnstock provides data during Vietnamese market hours
- **Holidays**: No data available during market holidays
- **Delisted Stocks**: Historical data may be incomplete
- **Corporate Actions**: Stock splits/dividends may affect historical prices

### Strategy Limitations
- **Transaction Costs**: Not included in backtesting (add ~0.15% per trade)
- **Slippage**: Real execution may differ from backtest prices
- **Market Impact**: Large orders can move prices
- **Survivorship Bias**: Only tests stocks that survived the period

### Risk Disclaimer
- **Past Performance**: Backtest results don't guarantee future returns
- **Market Conditions**: Strategies may perform differently in different market cycles
- **Fundamental Analysis**: Combine with company financial analysis
- **Risk Management**: Always use proper position sizing and stop losses

## 🛠️ Technical Details

### File Structure
```
finance-ai/
├── stragety-tester.py          # Main strategy tester class
├── demo_strategy_tester.py     # Demo and examples
├── requirements.txt            # Package dependencies
├── utils/
│   ├── trading_analysis.py     # Additional trading utilities
│   ├── data_processing.py      # Data processing helpers
│   └── plotting.py            # Visualization utilities
└── data/
    ├── analysis/              # Analysis results
    └── screenshots/           # Chart screenshots
```

### Architecture
- **Modular Design**: Easy to extend with new strategies
- **Data Pipeline**: Robust data fetching and processing
- **Error Handling**: Graceful handling of API failures
- **Performance Optimized**: Efficient calculations for large datasets

### Extending the Tool

Add new technical indicators:
```python
def add_stochastic(self, data):
    """Add Stochastic Oscillator"""
    df = data.copy()
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['%K'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(3).mean()
    return df
```

Add new strategies:
```python
def stochastic_strategy(self, data):
    """Stochastic Oscillator Strategy"""
    df = self.add_stochastic(data)
    df['signal'] = 0
    df.loc[df['%K'] < 20, 'signal'] = 1  # Buy oversold
    df.loc[df['%K'] > 80, 'signal'] = -1  # Sell overbought
    df['positions'] = df['signal'].diff()
    return df
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional trading strategies
- More technical indicators
- Portfolio optimization
- Machine learning integration
- Real-time trading signals
- Risk management tools

## 📄 License

This project is for educational purposes. Please use responsibly and do your own research before making investment decisions.

---

**Happy Trading! 📈**

*Remember: The best strategy is the one that fits your risk tolerance and investment goals.*
