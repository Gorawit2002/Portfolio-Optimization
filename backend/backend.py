from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
#import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import warnings
import time
import os
import pickle
from pathlib import Path
import json
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "https://portfolio-optimization-beige.vercel.app",  # Frontend URL ของคุณ
    "https://portfolio-optimization-beige.vercel.app/"  # เผื่อมี trailing slash
])

# Add these at the top of your file with other global variables
sp500_data = None
returns = None
benchmark_returns = None


def calculate_portfolio_metrics(selected_stocks, investment_horizon, rebalancing_freq, returns, risk_free_rate=0.03):
    """Calculate expected returns, volatilities, and Sharpe ratios for selected stocks"""
    expected_returns = {}
    volatilities = {}
    sharpe_ratios = {}
    market_return = returns.mean().mean() * 252

    for stock in selected_stocks:
        stock_returns = returns[stock]
        benchmark_returns = returns.mean(axis=1)

        # Investment horizon multipliers
        horizon_multiplier = {
            (1, 3): 1.2,    # Short-term
            (3, 10): 1.5,   # Medium-term
            (10, float('inf')): 2.0  # Long-term
        }

        # Get correct horizon multiplier
        current_horizon = [h for h in horizon_multiplier.keys()
                         if h[0] <= investment_horizon[1] <= h[1]][0]

        # Rebalancing frequency multipliers
        rebalance_multiplier = {
            12: 1.1,  # Monthly
            4: 1.05,  # Quarterly
            1: 1.0    # Annually
        }

        # Calculate beta using linear regression
        model = LinearRegression()
        model.fit(benchmark_returns.values.reshape(-1, 1), stock_returns.values)
        beta = model.coef_[0]

        # Calculate expected return using CAPM and multipliers
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        expected_return *= horizon_multiplier[current_horizon]
        expected_return *= rebalance_multiplier[rebalancing_freq]

        # Calculate volatility and Sharpe ratio
        vol = np.std(stock_returns) * np.sqrt(252)
        sharpe = (expected_return - risk_free_rate) / vol if vol > 0 else 0

        expected_returns[stock] = expected_return
        volatilities[stock] = vol
        sharpe_ratios[stock] = sharpe

    return expected_returns, volatilities, sharpe_ratios

def portfolio_optimization(expected_returns, cov_matrix, max_allocation=None, min_allocation=None):
    """Optimize portfolio weights using Mean-Variance Optimization"""
    num_assets = len(expected_returns)
    
    # Set allocation bounds
    bounds = [(min_allocation or 0.01, max_allocation or 1.0) for _ in range(num_assets)]

    # Portfolio constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
    ]

    # Objective function: Maximize Sharpe Ratio
    def objective(weights):
        port_return = np.dot(weights, list(expected_returns.values()))
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return / port_vol)  # Negative because we minimize

    # Initial guess of equal weights
    initial_weights = np.ones(num_assets) / num_assets

    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

def get_suggested_min_allocations(max_allocation, num_stocks):
    """Calculate suggested minimum allocations based on max allocation"""
    remaining = 1.0 - max_allocation
    suggestions = [
        remaining / (num_stocks - 1),  # Even distribution of remaining
        min(remaining / (num_stocks - 1), 0.1)  # Capped at 10%
    ]
    return suggestions

class MarketDataManager:
    def __init__(self, cache_dir='market_data_cache'):
        self.tickers = None
        self.price_data = None
        self.returns = None
        self.fundamental_data = None
        self.technical_indicators = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache file paths
        self.tickers_cache = self.cache_dir / 'tickers.json'
        self.price_cache = self.cache_dir / 'price_data.pkl'
        self.fundamentals_cache = self.cache_dir / 'fundamentals.pkl'
        self.cache_metadata = self.cache_dir / 'metadata.json'

    def _save_cache_metadata(self, last_update):
        metadata = {
            'last_update': last_update.strftime('%Y-%m-%d'),
        }
        with open(self.cache_metadata, 'w') as f:
            json.dump(metadata, f)

    def _load_cache_metadata(self):
        if self.cache_metadata.exists():
            with open(self.cache_metadata, 'r') as f:
                return json.load(f)
        return None

    def _is_cache_valid(self, max_age_days=1):
        metadata = self._load_cache_metadata()
        if not metadata:
            return False

        last_update = datetime.strptime(metadata['last_update'], '%Y-%m-%d')
        age = datetime.now() - last_update
        return age.days < max_age_days

    def fetch_sp500_tickers(self, use_cache=True):
        if use_cache and self.tickers_cache.exists():
            with open(self.tickers_cache, 'r') as f:
                self.tickers = json.load(f)
                return self.tickers

        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        self.tickers = sp500['Symbol'].tolist()

        # Save to cache
        with open(self.tickers_cache, 'w') as f:
            json.dump(self.tickers, f)

        return self.tickers

    def fetch_stock_data(self, start_date='2015-06-01', use_cache=True, force_refresh=False):
        try:
            if use_cache and not force_refresh and self._is_cache_valid():
                return self._load_cached_data()

            print("Fetching fresh market data...")
            if self.tickers is None:
                self.fetch_sp500_tickers()

            # Clean tickers
            self.tickers = [t.replace('.', '-') for t in self.tickers]

            current_date = datetime.today().strftime('%Y-%m-%d')
            data = yf.download(self.tickers, start=start_date, end=current_date, progress=False)

            # เปลี่ยนจาก 'Adj Close' เป็น 'Close'
            if 'Close' not in data.columns:
                raise ValueError("Price data missing 'Close' column. Data may be incomplete.")

            close_prices = data['Close']

            # Filter columns with enough data (90% non-NaN)
            self.price_data = close_prices.dropna(axis=1, thresh=len(close_prices)*0.9)
            self.returns = self.price_data.pct_change().dropna()

            # Get fundamentals
            self.fundamental_data = []
            valid_tickers = self.price_data.columns
            for ticker in valid_tickers:
                try:
                    fund_data = self._get_fundamental_data(ticker)
                    if fund_data and any(not pd.isna(v) for v in fund_data.values()):
                        self.fundamental_data.append(fund_data)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")

            # Save to cache
            self._save_to_cache()

            return self.price_data, self.returns, self.fundamental_data

        except Exception as e:
            print(f"Error: {str(e)}")
            return None, None, None

    def _get_fundamental_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'ticker': ticker,
                'pe_ratio': info.get('forwardPE', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'free_cash_flow': info.get('freeCashflow', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'profit_margins': info.get('profitMargins', np.nan),
                'dividend_yield': info.get('dividendYield', np.nan),
                'beta': info.get('beta', np.nan)
            }
        except:
            return None

    def _save_to_cache(self):
        # Save price data and returns
        with open(self.price_cache, 'wb') as f:
            pickle.dump({
                'price_data': self.price_data,
                'returns': self.returns
            }, f)

        # Save fundamental data
        with open(self.fundamentals_cache, 'wb') as f:
            pickle.dump(self.fundamental_data, f)

        # Save metadata
        self._save_cache_metadata(datetime.now())

    def _load_cached_data(self):
        print("Loading data from cache...")
        try:
            # Load price data and returns
            with open(self.price_cache, 'rb') as f:
                data = pickle.load(f)
                self.price_data = data['price_data']
                self.returns = data['returns']

            # Load fundamental data
            with open(self.fundamentals_cache, 'rb') as f:
                self.fundamental_data = pickle.load(f)

            return self.price_data, self.returns, self.fundamental_data

        except Exception as e:
            print(f"Error loading cache: {e}")
            return None, None, None

    def clear_cache(self):
        """Clear all cached data"""
        cache_files = [
            self.tickers_cache,
            self.price_cache,
            self.fundamentals_cache,
            self.cache_metadata
        ]
        for file in cache_files:
            if file.exists():
                file.unlink()
        print("Cache cleared successfully")

    def calculate_technical_indicators(self):
        if self.price_data is None:
            raise ValueError("Price data must be fetched first")

        def calculate_stock_indicators(prices):
            try:
                if isinstance(prices, pd.Series):
                    indicators = pd.DataFrame(index=prices.index)

                    # Basic indicators
                    indicators['SMA20'] = prices.rolling(window=20).mean()
                    indicators['SMA50'] = prices.rolling(window=50).mean()
                    indicators['ROC'] = (prices - prices.shift(10)) / prices.shift(10)
                    indicators['Daily_Return'] = prices.pct_change()
                    indicators['Volatility'] = indicators['Daily_Return'].rolling(window=20).std()

                    # Bollinger Bands
                    sma = prices.rolling(window=20).mean()
                    std = prices.rolling(window=20).std()
                    indicators['BB_Upper'] = sma + (std * 2)
                    indicators['BB_Lower'] = sma - (std * 2)
                    indicators['BB_Width'] = (indicators['BB_Upper'] - indicators['BB_Lower']) / sma

                    return indicators.dropna()
                return None
            except Exception as e:
                print(f"Error calculating indicators: {str(e)}")
                return None

        self.technical_indicators = {}
        for column in self.price_data.columns:
            try:
                self.technical_indicators[column] = calculate_stock_indicators(self.price_data[column])
            except Exception as e:
                print(f"Error processing {column}: {str(e)}")

        return self.technical_indicators

#old ml model
    # def train_ml_models(self):
    #     if not self.ml_features or len(self.ml_features) == 0:
    #         raise ValueError("ML features must be prepared before training")

    #     feature_df = pd.DataFrame(self.ml_features).T
    #     if feature_df.empty:
    #         raise ValueError("Feature DataFrame is empty. Cannot train models.")

    #     feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    #     feature_df = feature_df.fillna(feature_df.mean())

    #     X = self.scaler.fit_transform(feature_df)
    #     forward_returns = self.data_manager.returns.iloc[-20:].mean() * 252
    #     y = forward_returns.values

    #     self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    #     self.rf_model.fit(X, y)

    #     y_class = (y > y.mean()).astype(int)
    #     self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    #     self.gb_model.fit(X, y_class)

    #     self.kmeans = KMeans(n_clusters=5, random_state=42)
    #     self.clusters = self.kmeans.fit_predict(X)



class MLStockSelector:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.risk_free_rate = 0.03
        self.scaler = StandardScaler()
        self.ml_features = None
        self.rf_model = None
        self.gb_model = None
        self.kmeans = None

    def prepare_ml_features(self):
        if not self.data_manager.technical_indicators:
            raise ValueError("No technical indicators available.")
        if not self.data_manager.fundamental_data:
            raise ValueError("No fundamental data available.")

        fundamental_df = pd.DataFrame(self.data_manager.fundamental_data).set_index('ticker')
        self.ml_features = {}

        for ticker in self.data_manager.technical_indicators.keys():
            tech_features = self.data_manager.technical_indicators.get(ticker)
            if tech_features is None or tech_features.empty:
                continue

            fund_features = fundamental_df.loc[ticker] if ticker in fundamental_df.index else None
            if fund_features is None or fund_features.empty:
                continue

            returns_series = self.data_manager.returns[ticker].dropna()
            if len(returns_series) > 0:
                vol = returns_series.std() * np.sqrt(252)
                sharpe = (returns_series.mean() * 252 - self.risk_free_rate) / vol

                features = pd.concat([
                    tech_features.iloc[-1],
                    fund_features,
                    pd.Series({'volatility': vol, 'sharpe_ratio': sharpe})
                ])
                self.ml_features[ticker] = features

        if not self.ml_features:
            raise ValueError("No ML features prepared. Ensure data availability.")

    def train_ml_models(self):
        """
        Train RandomForest (regression), GradientBoostingClassifier (classification),
        and KMeans (clustering) using train-test split for model validation.
        """
        if not self.ml_features or len(self.ml_features) == 0:
            raise ValueError("ML features must be prepared before training.")

        # Prepare Features and Target
        feature_df = pd.DataFrame(self.ml_features).T
        if feature_df.empty:
            raise ValueError("Feature DataFrame is empty. Cannot train models.")

        # Handle NaNs and Infinite Values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(feature_df.mean())

        # Extract target (y) and align features (X)
        forward_returns = self.data_manager.returns.iloc[-20:].mean() * 252
        y = forward_returns.reindex(feature_df.index).dropna()
        X = feature_df.loc[y.index]  # Align features with the target

        # Scale the features
        X_scaled = self.scaler.fit_transform(X)

        # Train-Test Split(70/30)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.3, random_state=42)

        # Save X_test and y_test as class attributes for evaluation
        self.X_test = X_test
        self.y_test = y_test

        # Train RandomForest Regressor
        print("Training RandomForest Regressor...")
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        y_pred_rf = self.rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_rf)
        print(f"RandomForest Test MSE: {mse:.4f}")

        # Train GradientBoostingClassifier
        print("Training GradientBoostingClassifier...")
        y_class_train = (y_train > y_train.mean()).astype(int)
        y_class_test = (y_test > y_test.mean()).astype(int)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.gb_model.fit(X_train, y_class_train)
        y_pred_gb = self.gb_model.predict(X_test)

        # Classification Metrics
        accuracy = accuracy_score(y_class_test, y_pred_gb)
        precision = precision_score(y_class_test, y_pred_gb, zero_division=0)
        recall = recall_score(y_class_test, y_pred_gb, zero_division=0)
        f1 = f1_score(y_class_test, y_pred_gb, zero_division=0)
        print(f"GradientBoosting Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Train KMeans Clustering
        print("Training KMeans Clustering...")
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.clusters = self.kmeans.fit_predict(X_scaled)
        print("KMeans Clustering completed.")



    def select_stocks(self, num_stocks, min_vol=None, max_vol=None):
        feature_df = pd.DataFrame(self.ml_features).T
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(feature_df.mean())

        # Filter by volatility range
        if min_vol is not None:
            feature_df = feature_df[feature_df['volatility'] >= min_vol]
        if max_vol is not None:
            feature_df = feature_df[feature_df['volatility'] <= max_vol]

        if feature_df.empty:
            print("No stocks match volatility criteria")
            return []

        X = self.scaler.transform(feature_df)
        rf_predictions = self.rf_model.predict(X)
        gb_predictions = self.gb_model.predict_proba(X)[:, 1]
        cluster_predictions = self.kmeans.predict(X)

        scores = pd.DataFrame({
            'ticker': feature_df.index,
            'rf_score': rf_predictions,
            'gb_score': gb_predictions,
            'cluster': cluster_predictions,
            'volatility': feature_df['volatility']
        })

        scores['final_score'] = (scores['rf_score'].rank(pct=True) * 0.4 +
                               scores['gb_score'].rank(pct=True) * 0.4 +
                               scores['cluster'].rank(pct=True) * 0.2)

        return scores.nlargest(num_stocks, 'final_score')['ticker'].tolist()

def validate_and_process_horizon(horizon_choice):
    """
    Validates and processes investment horizon from API request.
    Matches existing horizon_map structure.
    """
    horizon_map = {
        'short': (1, 3),    # 1-3 years
        'medium': (3, 10),  # 3-10 years
        'long': (10, float('inf'))  # 10+ years
    }
    
    if horizon_choice not in horizon_map:
        raise ValueError(f"Invalid investment horizon. Must be one of: {', '.join(horizon_map.keys())}")
    
    return horizon_map[horizon_choice]

def calculate_allocation_suggestions(max_allocation, num_stocks):
    """
    Calculates suggested minimum allocations based on max allocation.
    Uses the same logic as your get_suggested_min_allocations function.
    """
    if not (0 < max_allocation <= 0.5):
        raise ValueError("Maximum allocation must be between 0 and 0.5 (50%)")
    
    remaining = 1.0 - max_allocation
    suggestions = [
        remaining / (num_stocks - 1),  # Even distribution
        min(remaining / (num_stocks - 1), 0.1)  # Capped at 10%
    ]
    
    return {
        "suggestions": [
            {
                "value": suggestions[0],
                "description": "Even distribution of remaining allocation"
            },
            {
                "value": suggestions[1],
                "description": "Capped at 10% per stock"
            }
        ]
    }

def validate_volatility_range(min_vol, max_vol):
    """
    Validates volatility range inputs from API request.
    Matches your existing volatility validation logic.
    """
    if min_vol is not None:
        min_vol = float(min_vol)
        if min_vol > 1:  # Convert from percentage
            min_vol = min_vol / 100
            
    if max_vol is not None:
        max_vol = float(max_vol)
        if max_vol > 1:  # Convert from percentage
            max_vol = max_vol / 100

    # Validation using your existing logic
    if min_vol is not None and max_vol is not None:
        if min_vol >= max_vol:
            raise ValueError("Minimum volatility must be less than maximum volatility")
        if min_vol < 0 or max_vol < 0:
            raise ValueError("Volatility cannot be negative")
        if max_vol > 1:
            raise ValueError("Maximum volatility cannot exceed 100%")
            
    return min_vol, max_vol

def process_rebalancing_frequency(freq_choice):
    """
    Processes rebalancing frequency from API request.
    Matches your existing rebalance_map structure.
    """
    rebalance_map = {
        'monthly': 12,
        'quarterly': 4,
        'annually': 1
    }
    
    if freq_choice not in rebalance_map:
        raise ValueError(f"Invalid rebalancing frequency. Must be one of: {', '.join(rebalance_map.keys())}")
        
    return rebalance_map[freq_choice]

def format_portfolio_results(selected_stocks, expected_returns, volatilities, 
                           sharpe_ratios, optimal_weights, cov_matrix, balance):
    """
    Formats portfolio optimization results for API response.
    Matches your existing allocation response structure.
    """
    allocations = []
    for i, stock in enumerate(selected_stocks):
        allocation = {
            "symbol": stock,
            "security": stock,  # Will be updated with company name in the optimize route
            "allocation": round(optimal_weights[i] * balance, 2),
            "expected_return": round(expected_returns[stock], 4),
            "volatility": round(volatilities[stock], 4),
            "sharpe_ratio": round(sharpe_ratios[stock], 4),
            "weight": float(optimal_weights[i])
        }
        allocations.append(allocation)

    return {"allocations": allocations}


#data_manager = MarketDataManager()
#price_data, returns, fundamental_data = data_manager.fetch_stock_data(use_cache=True)       
#data_manager.calculate_technical_indicators()

@app.route('/api/optimize', methods=['POST'])
def optimize():
    data = request.json
    print("Received optimization request:", data)

    try:
        # Validate basic parameters
        mode = data.get('mode')
        balance = float(data.get('balance', 0))
        num_stocks = int(data.get('shares', 2))

        if balance < 1000:
            return jsonify({"error": "Minimum investment amount is $1,000"}), 400
        if not 2 <= num_stocks <= 10:
            return jsonify({"error": "Number of stocks must be between 2 and 10"}), 400

        # Process advanced parameters
        advanced_params = data.get('advancedParams', {})
        
        # Get investment horizon
        investment_horizon = validate_and_process_horizon(
            advanced_params.get('investmentHorizon', 'short')
        )
        
        # Get rebalancing frequency
        rebalancing_freq = process_rebalancing_frequency(
            advanced_params.get('rebalanceFrequency', 'monthly')
        )
        
        # Process volatility constraints
        min_volatility, max_volatility = validate_volatility_range(
            advanced_params.get('minVolatility'),
            advanced_params.get('maxVolatility')
        )
        
        # Process allocation constraints
        max_allocation = advanced_params.get('maxAllocation')
        min_allocation = advanced_params.get('minAllocation')
        
        if max_allocation:
            max_allocation = float(max_allocation) / 100
            # Calculate suggested min allocations if max is provided
            suggestions = calculate_allocation_suggestions(max_allocation, num_stocks)
            # Store suggestions in session if needed
            # session['allocation_suggestions'] = suggestions
        
        if min_allocation:
            min_allocation = float(min_allocation) / 100

        # Initialize data manager and prepare data
        data_manager = MarketDataManager()
        price_data, returns, fundamental_data = data_manager.fetch_stock_data(use_cache=True)
        data_manager.calculate_technical_indicators()

        selected_stocks = []
        if mode == 'auto':
            # Use ML for stock selection
            selector = MLStockSelector(data_manager)
            selector.prepare_ml_features()
            selector.train_ml_models()
            selected_stocks = selector.select_stocks(
                num_stocks,
                min_volatility,
                max_volatility
            )
        else:
            # Manual mode - use provided stocks
            selected_stocks = data.get('stocks', [])

        if not selected_stocks:
            return jsonify({"error": "No valid stocks selected"}), 400

        # Calculate portfolio metrics
        expected_returns, volatilities, sharpe_ratios = calculate_portfolio_metrics(
            selected_stocks,
            investment_horizon,
            rebalancing_freq,
            returns
        )

        # Portfolio optimization
        cov_matrix = returns[selected_stocks].cov() * 252
        optimal_weights = portfolio_optimization(
            expected_returns,
            cov_matrix,
            max_allocation,
            min_allocation
        )

        results = format_portfolio_results(
                selected_stocks,
                expected_returns,
                volatilities,
                sharpe_ratios,
                optimal_weights,
                cov_matrix,
                balance
            )
        
# Add allocation suggestions if available
        if 'suggestions' in locals():
            results["allocation_suggestions"] = suggestions

        print("Final results being sent:", results)  # Add this line
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in optimization: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    try:
        data_manager = MarketDataManager()
        tickers_with_info = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        stocks_list = [
            {
                'Symbol': row['Symbol'],
                'Security': row['Security']
            }
            for _, row in tickers_with_info.iterrows()
        ]
        return jsonify(stocks_list)
    except Exception as e:
        print(f"Error fetching stocks: {e}")
        return jsonify({"error": "Failed to load stock data"}), 500
    
@app.route('/api/stock-data', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol')
    period = request.args.get('period', '1d')  # Default to 6 months if not specified
    
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    try:
        # Fetch stock data
        stock = yf.Ticker(symbol)
        
        # Get interval based on period
        interval = '1d'  # default interval
        if period == '1d':
            interval = '1m'
            hist = stock.history(period='1d', interval='1m', prepost=True)
        elif period == '5d':
            interval = '1m'  # Changed from 5m to 1m for more frequent updates
            hist = stock.history(period='5d', interval='1m', prepost=True)
        else:
            hist = stock.history(period=period, interval=interval)
        
        # Format the data
        data = []
        for date, row in hist.iterrows():
            data.append({
                'date': date.strftime('%Y-%m-%d %H:%M:%S') if interval.endswith('m') else date.strftime('%Y-%m-%d'),
                'close': float(row['Close']),
                'volume': int(row['Volume']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'open': float(row['Open'])
            })
        
        # Calculate percentage change
        if len(data) > 0:
            current_price = data[-1]['close']
            prev_price = data[-2]['close'] if len(data) > 1 else data[0]['close']
            percent_change = ((current_price - prev_price) / prev_price * 100)
        else:
            current_price = 0
            percent_change = 0

        return jsonify({
            'data': data,
            'currentPrice': current_price,
            'percentChange': percent_change,
            'lastUpdate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return jsonify({"error": "Failed to fetch stock data"}), 500

@app.route('/api/sp500-details', methods=['GET'])
def get_sp500_details():
    try:
        # Fetch S&P 500 data using '^GSPC' symbol
        sp500 = yf.Ticker('^GSPC')
        
        # Get current data
        info = sp500.info

        # Get historical daily data
        daily_data = sp500.history(period='5d')  # Using 5d to ensure we have enough data
        
        # Get today's minute data for current price
        today_data = sp500.history(period='1d', interval='1m', prepost=True)
        
        if len(daily_data) > 1 and len(today_data) > 0:
            current_price = today_data['Close'].iloc[-1]
            previous_close = daily_data['Close'].iloc[-2]  # Get the second-to-last daily close
            
            print("Current Price:", current_price)
            print("Previous Close:", previous_close)
            
            # Calculate day range from today's data
            day_low = today_data['Low'].min()
            day_high = today_data['High'].max()
            
            # Get year data for year range
            year_data = sp500.history(period='1y')
            year_low = year_data['Low'].min()
            year_high = year_data['High'].max()
            
            # Calculate percentage change with more precision
            percent_change = ((current_price - previous_close) / previous_close) * 100
            # Use numpy for more precise formatting
            formatted_percent_change = current_price - previous_close
            print("Raw Percent Change:", percent_change)
            print("Percent Change:", formatted_percent_change)
            
            market_cap = float(current_price) * 8.8e9 / 1e12
            print("Calculated Market Cap:", market_cap)

            data = {
                'previousClose': float(previous_close),
                'currentPrice': float(current_price),
                'dayRange': {
                    'low': float(day_low),
                    'high': float(day_high)
                },
                'yearRange': {
                    'low': float(year_low),
                    'high': float(year_high)
                },
                'marketCap': market_cap,
                'volume': info.get('volume', 0),
                'percentChange': formatted_percent_change,
                'lastUpdate': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return jsonify(data)
        else:
            raise Exception("Insufficient data available")
            
    except Exception as e:
        print(f"Error fetching S&P 500 details: {e}")
        return jsonify({"error": "Failed to fetch S&P 500 data"}), 500
    
@app.route('/api/real-time-price', methods=['GET'])
def get_real_time_price():
    symbol = request.args.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    try:
        stock = yf.Ticker(symbol)
        latest = stock.history(period='1d', interval='1m', prepost=True)
        
        if not latest.empty:
            current_price = float(latest['Close'].iloc[-1])
            previous_price = float(latest['Close'].iloc[-2]) if len(latest) > 1 else current_price
            percent_change = ((current_price - previous_price) / previous_price * 100)
            
            return jsonify({
                'symbol': symbol,
                'price': current_price,
                'change': percent_change,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            return jsonify({"error": "No data available"}), 404

    except Exception as e:
        print(f"Error fetching real-time price: {e}")
        return jsonify({"error": "Failed to fetch real-time price"}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)