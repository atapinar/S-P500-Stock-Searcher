import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class StockAnalyzer:
    def __init__(self, author_name):
        self.author_name = author_name
        self.output_dir = 'stock_analysis_results'
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_user_input(self):
        """Get stock tickers and date range from user."""
        print(f"\n=== Stock Analysis Configuration by {self.author_name} ===")
        
        # Get stock tickers
        while True:
            tickers = input("Enter stock tickers separated by comma (e.g., GOOGL,AAPL,MSFT): ").strip()
            if tickers:
                tickers = [tick.strip().upper() for tick in tickers.split(',')]
                break
            print("Please enter at least one ticker.")
        
        # Get date range
        while True:
            try:
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                datetime.strptime(start_date, '%Y-%m-%d')
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD")
        
        while True:
            try:
                end_date = input("Enter end date (YYYY-MM-DD): ").strip()
                datetime.strptime(end_date, '%Y-%m-%d')
                break
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD")
        
        # Get volatility window
        while True:
            try:
                vol_window = int(input("Enter number of days for volatility calculation (e.g., 30): "))
                if vol_window > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get RSI period
        while True:
            try:
                rsi_period = int(input("Enter number of days for RSI calculation (e.g., 14): "))
                if rsi_period > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        return tickers, start_date, end_date, vol_window, rsi_period

    def calculate_rsi(self, data, periods=14):
        """Calculate RSI for the given data."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def download_stock_data(self, ticker, start_date, end_date, lookback_days):
        """Download stock data for a given ticker with additional lookback period."""
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            lookback_start = (start_dt - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            data = yf.download(ticker, start=lookback_start, end=end_date)
            return data
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
            return None

    def calculate_metrics(self, data, vol_window, rsi_period, start_date):
        """Calculate log returns, volatility and RSI for the stock data."""
        data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Log_Ret'].rolling(window=vol_window).std() * np.sqrt(252)
        data['RSI'] = self.calculate_rsi(data, rsi_period)
        
        # Add author attribution
        data.attrs['author'] = self.author_name
        
        if start_date:
            data = data[start_date:]
        return data

    def plot_stock_analysis(self, stock_data, ticker, vol_window, rsi_period):
        """Create plots for a single stock."""
        # Create figure with extra height for the author name
        fig = plt.figure(figsize=(12, 13))
        
        # Create GridSpec for better control of spacing
        gs = fig.add_gridspec(4, 1, height_ratios=[0.2, 1, 1, 1], hspace=0.4)
        
        # Add main title in its own subplot
        title_ax = fig.add_subplot(gs[0])
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, f'{ticker} Stock Analysis', 
                     fontsize=16, ha='center', va='center')
        
        # Create other subplots
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])
        
        # Plot Close price
        stock_data['Close'].plot(ax=ax1, color='blue')
        ax1.set_title('Close Price')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True)
        
        # Plot Volatility
        stock_data['Volatility'].plot(ax=ax2, color='red')
        ax2.set_title(f'{vol_window}-Day Rolling Volatility')
        ax2.set_ylabel('Volatility')
        ax2.grid(True)
        
        # Plot RSI
        stock_data['RSI'].plot(ax=ax3, color='purple')
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax3.set_title(f'{rsi_period}-Day RSI')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True)
        
        # Add author signature in right corner
        fig.text(0.95, 0.02, f'{self.author_name}', 
                fontsize=12, color='black', ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=5))
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.1, top=0.95)
        return fig

    def save_analysis(self, stock_data, ticker, fig):
        """Save the analysis results."""
        # Create ticker-specific directory
        ticker_dir = os.path.join(self.output_dir, ticker)
        if not os.path.exists(ticker_dir):
            os.makedirs(ticker_dir)
        
        # Save the plot
        fig.savefig(os.path.join(ticker_dir, f'{ticker}_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        
        # Save the data to CSV with metadata
        csv_path = os.path.join(ticker_dir, f'{ticker}_data.csv')
        stock_data.to_csv(csv_path)
        
        # Save metadata
        metadata = {
            'Author': self.author_name,
            'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ticker': ticker,
            'Data Range': f"{stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}"
        }
        
        with open(os.path.join(ticker_dir, 'metadata.txt'), 'w') as f:
            for key, value in metadata.items():
                f.write(f'{key}: {value}\n')

    def run_analysis(self):
        """Run the complete analysis."""
        stocks, start_date, end_date, vol_window, rsi_period = self.get_user_input()
        lookback_days = max(vol_window, rsi_period) * 2
        stock_data_dict = {}
        
        for ticker in stocks:
            print(f"\nProcessing {ticker}...")
            
            data = self.download_stock_data(ticker, start_date, end_date, lookback_days)
            if data is not None:
                data = self.calculate_metrics(data, vol_window, rsi_period, start_date)
                stock_data_dict[ticker] = data
                
                print(f"Shape of data: {data.shape}")
                print("\nLast 5 days of closing prices:")
                print(data['Close'].tail())
                print(f"\nCurrent Volatility: {data['Volatility'].iloc[-1]:.4f}")
                print(f"Current RSI: {data['RSI'].iloc[-1]:.2f}")
                
                fig = self.plot_stock_analysis(data, ticker, vol_window, rsi_period)
                self.save_analysis(data, ticker, fig)
                plt.show()
        
        return stock_data_dict

def main():
    author_name = input("Please enter your name: ").strip()
    analyzer = StockAnalyzer(author_name)
    stock_data = analyzer.run_analysis()
    return stock_data

if __name__ == "__main__":
    stock_data = main()
