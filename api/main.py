from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import json
import uuid
import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
import sys
try:
    from spoon_ai.memory.short_term_manager import ShortTermMemoryManager
    from spoon_ai.schema import Message
except Exception:
    ShortTermMemoryManager = None
    Message = None
import asyncio
import random
import time
import yfinance as yf
import pandas as pd
import numpy as np

# Import SpoonAI agents
from agents import (
    create_trading_orchestrator_agent,
    create_scout_agent,
    create_evaluation_agent,
    create_sentiment_agent,
    create_ranking_agent,
    TradingState
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
def configure_logging():
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.DEBUG)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).setLevel(logging.DEBUG)
configure_logging()
app = FastAPI(title="Just Buy It API", version="1.0.0")
INMEM_STOCK_DATA: Dict[str, List[Dict[str, Any]]] = {}

# Initialize SpoonAI agents
trading_orchestrator = None
scout_agent = None
evaluation_agent = None
sentiment_agent = None
ranking_agent = None

def initialize_agents():
    """Initialize SpoonAI agents"""
    global trading_orchestrator, scout_agent, evaluation_agent, sentiment_agent, ranking_agent
    
    try:
        trading_orchestrator = create_trading_orchestrator_agent()
        scout_agent = create_scout_agent()
        evaluation_agent = create_evaluation_agent()
        sentiment_agent = create_sentiment_agent()
        ranking_agent = create_ranking_agent()
        print("✅ SpoonAI agents initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing SpoonAI agents: {e}")
        # Fallback to None - agents will be created on demand

# Initialize agents on startup
initialize_agents()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

def preload_stock_data():
    global INMEM_STOCK_DATA
    try:
        INMEM_STOCK_DATA = get_stock_data()
    except Exception:
        INMEM_STOCK_DATA = {}

preload_stock_data()

MEMORY_THREADS: Dict[str, List[Any]] = {}
MEMORY_MANAGERS: Dict[str, Any] = {}

def _ensure_thread(thread_id: str):
    if thread_id not in MEMORY_THREADS:
        MEMORY_THREADS[thread_id] = []
    if thread_id not in MEMORY_MANAGERS:
        MEMORY_MANAGERS[thread_id] = ShortTermMemoryManager() if ShortTermMemoryManager else None

# Complete S&P 500 companies with sectors
SP500_STOCKS = [
    {'ticker': 'SPY', 'sector': 'Index'},
    {'ticker': 'AAPL', 'sector': 'Technology'},
    {'ticker': 'MSFT', 'sector': 'Technology'},
    {'ticker': 'GOOGL', 'sector': 'Technology'},
    {'ticker': 'AMZN', 'sector': 'Consumer Discretionary'},
    {'ticker': 'NVDA', 'sector': 'Technology'},
    {'ticker': 'TSLA', 'sector': 'Consumer Discretionary'},
    {'ticker': 'META', 'sector': 'Technology'},
    {'ticker': 'BRK-B', 'sector': 'Financials'},
    {'ticker': 'UNH', 'sector': 'Healthcare'},
    {'ticker': 'JNJ', 'sector': 'Healthcare'},
    {'ticker': 'XOM', 'sector': 'Energy'},
    {'ticker': 'JPM', 'sector': 'Financials'},
    {'ticker': 'V', 'sector': 'Financials'},
    {'ticker': 'PG', 'sector': 'Consumer Staples'},
    {'ticker': 'HD', 'sector': 'Consumer Discretionary'},
    {'ticker': 'MA', 'sector': 'Financials'},
    {'ticker': 'CVX', 'sector': 'Energy'},
    {'ticker': 'LLY', 'sector': 'Healthcare'},
    {'ticker': 'ABBV', 'sector': 'Healthcare'},
    {'ticker': 'PFE', 'sector': 'Healthcare'},
    {'ticker': 'KO', 'sector': 'Consumer Staples'},
    {'ticker': 'PEP', 'sector': 'Consumer Staples'},
    {'ticker': 'WMT', 'sector': 'Consumer Staples'},
    {'ticker': 'MRK', 'sector': 'Healthcare'},
    {'ticker': 'BAC', 'sector': 'Financials'},
    {'ticker': 'ADBE', 'sector': 'Technology'},
    {'ticker': 'NFLX', 'sector': 'Communication Services'},
    {'ticker': 'CRM', 'sector': 'Technology'},
    {'ticker': 'ACN', 'sector': 'Information Technology'},
    {'ticker': 'TMO', 'sector': 'Healthcare'},
    {'ticker': 'COST', 'sector': 'Consumer Staples'},
    {'ticker': 'AVGO', 'sector': 'Technology'},
    {'ticker': 'ABT', 'sector': 'Healthcare'},
    {'ticker': 'DHR', 'sector': 'Healthcare'},
    {'ticker': 'CMCSA', 'sector': 'Communication Services'},
    {'ticker': 'DIS', 'sector': 'Communication Services'},
    {'ticker': 'VZ', 'sector': 'Communication Services'},
    {'ticker': 'NEE', 'sector': 'Utilities'},
    {'ticker': 'TXN', 'sector': 'Technology'},
    {'ticker': 'BMY', 'sector': 'Healthcare'},
    {'ticker': 'QCOM', 'sector': 'Technology'},
    {'ticker': 'UPS', 'sector': 'Industrials'},
    {'ticker': 'LOW', 'sector': 'Consumer Discretionary'},
    {'ticker': 'AMGN', 'sector': 'Healthcare'},
    {'ticker': 'SPGI', 'sector': 'Financials'},
    {'ticker': 'HON', 'sector': 'Industrials'},
    {'ticker': 'LIN', 'sector': 'Materials'},
    {'ticker': 'SBUX', 'sector': 'Consumer Discretionary'},
    {'ticker': 'INTU', 'sector': 'Technology'},
    {'ticker': 'RTX', 'sector': 'Industrials'},
    {'ticker': 'AMD', 'sector': 'Technology'},
    {'ticker': 'AMT', 'sector': 'Real Estate'},
    {'ticker': 'CAT', 'sector': 'Industrials'},
    {'ticker': 'GS', 'sector': 'Financials'},
    {'ticker': 'CSCO', 'sector': 'Technology'},
    {'ticker': 'ISRG', 'sector': 'Healthcare'},
    {'ticker': 'BKNG', 'sector': 'Consumer Discretionary'},
    {'ticker': 'MDT', 'sector': 'Healthcare'},
    {'ticker': 'TJX', 'sector': 'Consumer Discretionary'},
    {'ticker': 'COP', 'sector': 'Energy'},
    {'ticker': 'AXP', 'sector': 'Financials'},
    {'ticker': 'ADP', 'sector': 'Industrials'},
    {'ticker': 'PLD', 'sector': 'Real Estate'},
    {'ticker': 'DE', 'sector': 'Industrials'},
    {'ticker': 'SYK', 'sector': 'Healthcare'},
    {'ticker': 'NOW', 'sector': 'Technology'},
    {'ticker': 'GE', 'sector': 'Industrials'},
    {'ticker': 'TMUS', 'sector': 'Communication Services'},
    {'ticker': 'MS', 'sector': 'Financials'},
    {'ticker': 'EL', 'sector': 'Consumer Staples'},
    {'ticker': 'FISV', 'sector': 'Technology'},
    {'ticker': 'ORCL', 'sector': 'Technology'},
    {'ticker': 'CI', 'sector': 'Healthcare'},
    {'ticker': 'NKE', 'sector': 'Consumer Discretionary'},
    {'ticker': 'BLK', 'sector': 'Financials'},
    {'ticker': 'MU', 'sector': 'Technology'},
    {'ticker': 'MMM', 'sector': 'Industrials'},
    {'ticker': 'LMT', 'sector': 'Industrials'},
    {'ticker': 'ITW', 'sector': 'Industrials'},
    {'ticker': 'USB', 'sector': 'Financials'},
    {'ticker': 'TFC', 'sector': 'Financials'},
    {'ticker': 'NOC', 'sector': 'Industrials'},
    {'ticker': 'IBM', 'sector': 'Technology'},
    {'ticker': 'C', 'sector': 'Financials'},
    {'ticker': 'CL', 'sector': 'Consumer Staples'},
    {'ticker': 'MO', 'sector': 'Consumer Staples'},
    {'ticker': 'ETN', 'sector': 'Industrials'},
    {'ticker': 'SO', 'sector': 'Utilities'},
    {'ticker': 'MDLZ', 'sector': 'Consumer Staples'},
    {'ticker': 'GILD', 'sector': 'Healthcare'},
    {'ticker': 'WM', 'sector': 'Industrials'},
    {'ticker': 'DUK', 'sector': 'Utilities'},
    {'ticker': 'REGN', 'sector': 'Healthcare'},
    {'ticker': 'GD', 'sector': 'Industrials'},
    {'ticker': 'ZTS', 'sector': 'Healthcare'},
    {'ticker': 'APD', 'sector': 'Materials'},
    {'ticker': 'CSX', 'sector': 'Industrials'},
    {'ticker': 'BDX', 'sector': 'Healthcare'},
    {'ticker': 'PNC', 'sector': 'Financials'},
    {'ticker': 'CB', 'sector': 'Financials'},
    {'ticker': 'CME', 'sector': 'Financials'},
    {'ticker': 'T', 'sector': 'Communication Services'},
    {'ticker': 'CHTR', 'sector': 'Communication Services'},
    {'ticker': 'CCI', 'sector': 'Real Estate'},
    {'ticker': 'ATVI', 'sector': 'Communication Services'},
    {'ticker': 'BSX', 'sector': 'Healthcare'},
    {'ticker': 'ICE', 'sector': 'Financials'},
    {'ticker': 'EQIX', 'sector': 'Real Estate'},
    {'ticker': 'AON', 'sector': 'Financials'},
    {'ticker': 'PGR', 'sector': 'Financials'},
    {'ticker': 'ECL', 'sector': 'Materials'},
    {'ticker': 'MET', 'sector': 'Financials'},
    {'ticker': 'AEP', 'sector': 'Utilities'},
    {'ticker': 'PSX', 'sector': 'Energy'},
    {'ticker': 'SCHW', 'sector': 'Financials'},
    {'ticker': 'PH', 'sector': 'Industrials'},
    {'ticker': 'FCX', 'sector': 'Materials'},
    {'ticker': 'HUM', 'sector': 'Healthcare'},
    {'ticker': 'PXD', 'sector': 'Energy'},
    {'ticker': 'SHW', 'sector': 'Materials'},
    {'ticker': 'D', 'sector': 'Utilities'},
    {'ticker': 'F', 'sector': 'Consumer Discretionary'},
    {'ticker': 'IT', 'sector': 'Technology'},
    {'ticker': 'NEM', 'sector': 'Materials'},
    {'ticker': 'WFC', 'sector': 'Financials'},
    {'ticker': 'DG', 'sector': 'Consumer Discretionary'},
    {'ticker': 'TRV', 'sector': 'Financials'},
    {'ticker': 'KMB', 'sector': 'Consumer Staples'},
    {'ticker': 'SRE', 'sector': 'Utilities'},
    {'ticker': 'EOG', 'sector': 'Energy'},
    {'ticker': 'EW', 'sector': 'Healthcare'},
    {'ticker': 'KLAC', 'sector': 'Technology'},
    {'ticker': 'OXY', 'sector': 'Energy'},
    {'ticker': 'KHC', 'sector': 'Consumer Staples'},
    {'ticker': 'AFL', 'sector': 'Financials'},
    {'ticker': 'MCD', 'sector': 'Consumer Discretionary'},
    {'ticker': 'KMI', 'sector': 'Energy'},
    {'ticker': 'MAR', 'sector': 'Consumer Discretionary'},
    {'ticker': 'STZ', 'sector': 'Consumer Staples'},
    {'ticker': 'VLO', 'sector': 'Energy'},
    {'ticker': 'CTAS', 'sector': 'Industrials'},
    {'ticker': 'COF', 'sector': 'Financials'},
    {'ticker': 'FTNT', 'sector': 'Technology'},
    {'ticker': 'TT', 'sector': 'Industrials'},
    {'ticker': 'JCI', 'sector': 'Industrials'},
    {'ticker': 'DLR', 'sector': 'Real Estate'},
    {'ticker': 'AIG', 'sector': 'Financials'},
    {'ticker': 'ALL', 'sector': 'Financials'},
    {'ticker': 'KEYS', 'sector': 'Technology'},
    {'ticker': 'HCA', 'sector': 'Healthcare'},
    {'ticker': 'PCAR', 'sector': 'Industrials'},
    {'ticker': 'ROST', 'sector': 'Consumer Discretionary'},
    {'ticker': 'OTIS', 'sector': 'Industrials'},
    {'ticker': 'ED', 'sector': 'Utilities'},
    {'ticker': 'HPQ', 'sector': 'Technology'},
    {'ticker': 'WELL', 'sector': 'Real Estate'},
    {'ticker': 'HAL', 'sector': 'Energy'},
    {'ticker': 'IDXX', 'sector': 'Healthcare'},
    {'ticker': 'RMD', 'sector': 'Healthcare'},
    {'ticker': 'SLB', 'sector': 'Energy'},
    {'ticker': 'PEG', 'sector': 'Utilities'},
    {'ticker': 'DOW', 'sector': 'Materials'},
    {'ticker': 'FAST', 'sector': 'Industrials'},
    {'ticker': 'XYL', 'sector': 'Industrials'},
    {'ticker': 'CNC', 'sector': 'Healthcare'},
    {'ticker': 'EXC', 'sector': 'Utilities'},
    {'ticker': 'CTVA', 'sector': 'Materials'},
    {'ticker': 'ALGN', 'sector': 'Healthcare'},
    {'ticker': 'ES', 'sector': 'Utilities'},
    {'ticker': 'LHX', 'sector': 'Industrials'},
    {'ticker': 'FANG', 'sector': 'Energy'},
    {'ticker': 'CARR', 'sector': 'Industrials'},
    {'ticker': 'WBD', 'sector': 'Communication Services'},
    {'ticker': 'DLTR', 'sector': 'Consumer Discretionary'},
    {'ticker': 'WEC', 'sector': 'Utilities'},
    {'ticker': 'KR', 'sector': 'Consumer Staples'},
    {'ticker': 'CDNS', 'sector': 'Technology'},
    {'ticker': 'FITB', 'sector': 'Financials'},
    {'ticker': 'DVN', 'sector': 'Energy'},
    {'ticker': 'CAH', 'sector': 'Healthcare'},
    {'ticker': 'RSG', 'sector': 'Industrials'},
    {'ticker': 'ANET', 'sector': 'Technology'},
    {'ticker': 'ALB', 'sector': 'Materials'},
    {'ticker': 'CPRT', 'sector': 'Industrials'},
    {'ticker': 'ARE', 'sector': 'Real Estate'},
    {'ticker': 'TDG', 'sector': 'Industrials'},
    {'ticker': 'ODFL', 'sector': 'Industrials'},
    {'ticker': 'BIIB', 'sector': 'Healthcare'},
    {'ticker': 'SNPS', 'sector': 'Technology'},
    {'ticker': 'VICI', 'sector': 'Real Estate'},
    {'ticker': 'ESS', 'sector': 'Real Estate'},
    {'ticker': 'HES', 'sector': 'Energy'},
    {'ticker': 'HIG', 'sector': 'Financials'},
    {'ticker': 'AMCR', 'sector': 'Materials'},
    {'ticker': 'CCL', 'sector': 'Consumer Discretionary'},
    {'ticker': 'HBAN', 'sector': 'Financials'},
    {'ticker': 'CMI', 'sector': 'Industrials'},
    {'ticker': 'AES', 'sector': 'Utilities'},
    {'ticker': 'RCL', 'sector': 'Consumer Discretionary'},
    {'ticker': 'FLT', 'sector': 'Technology'},
    {'ticker': 'GPC', 'sector': 'Consumer Discretionary'},
    {'ticker': 'VTRS', 'sector': 'Healthcare'},
    {'ticker': 'IRM', 'sector': 'Real Estate'},
    {'ticker': 'INFO', 'sector': 'Industrials'},
    {'ticker': 'PARA', 'sector': 'Communication Services'},
    {'ticker': 'BALL', 'sector': 'Materials'},
    {'ticker': 'MTB', 'sector': 'Financials'},
    {'ticker': 'AKAM', 'sector': 'Technology'},
    {'ticker': 'PPG', 'sector': 'Materials'},
    {'ticker': 'SYF', 'sector': 'Financials'},
    {'ticker': 'BAX', 'sector': 'Healthcare'},
    {'ticker': 'NDAQ', 'sector': 'Financials'},
    {'ticker': 'LH', 'sector': 'Healthcare'},
    {'ticker': 'EFX', 'sector': 'Industrials'},
    {'ticker': 'TSN', 'sector': 'Consumer Staples'},
    {'ticker': 'TRMB', 'sector': 'Technology'},
    {'ticker': 'CF', 'sector': 'Materials'},
    {'ticker': 'AVB', 'sector': 'Real Estate'},
    {'ticker': 'BR', 'sector': 'Financials'},
    {'ticker': 'TROW', 'sector': 'Financials'},
    {'ticker': 'MAA', 'sector': 'Real Estate'},
    {'ticker': 'PPL', 'sector': 'Utilities'},
    {'ticker': 'PRU', 'sector': 'Financials'},
    {'ticker': 'MRO', 'sector': 'Energy'},
    {'ticker': 'APTV', 'sector': 'Consumer Discretionary'},
    {'ticker': 'FMC', 'sector': 'Materials'},
    {'ticker': 'EQR', 'sector': 'Real Estate'},
    {'ticker': 'EXPE', 'sector': 'Consumer Discretionary'},
    {'ticker': 'MOS', 'sector': 'Materials'},
    {'ticker': 'CNP', 'sector': 'Utilities'},
    {'ticker': 'XEL', 'sector': 'Utilities'},
    {'ticker': 'CMS', 'sector': 'Utilities'},
    {'ticker': 'BEN', 'sector': 'Financials'},
    {'ticker': 'IP', 'sector': 'Materials'},
    {'ticker': 'NRG', 'sector': 'Utilities'},
    {'ticker': 'NCLH', 'sector': 'Consumer Discretionary'},
    {'ticker': 'RE', 'sector': 'Financials'},
    {'ticker': 'NVR', 'sector': 'Consumer Discretionary'},
    {'ticker': 'EPAM', 'sector': 'Technology'},
    {'ticker': 'HAS', 'sector': 'Consumer Discretionary'},
    {'ticker': 'VMC', 'sector': 'Materials'},
    {'ticker': 'RHI', 'sector': 'Industrials'},
    {'ticker': 'LUV', 'sector': 'Industrials'},
    {'ticker': 'WRB', 'sector': 'Financials'},
    {'ticker': 'SWK', 'sector': 'Industrials'},
    {'ticker': 'DISH', 'sector': 'Communication Services'},
    {'ticker': 'J', 'sector': 'Industrials'},
    {'ticker': 'CPB', 'sector': 'Consumer Staples'},
    {'ticker': 'SJM', 'sector': 'Consumer Staples'},
    {'ticker': 'IRM', 'sector': 'Real Estate'},
    {'ticker': 'AIZ', 'sector': 'Financials'},
    {'ticker': 'CMA', 'sector': 'Financials'},
    {'ticker': 'MHK', 'sector': 'Consumer Discretionary'},
    {'ticker': 'LEG', 'sector': 'Consumer Discretionary'},
    {'ticker': 'PBCT', 'sector': 'Financials'},
    {'ticker': 'RL', 'sector': 'Consumer Discretionary'},
    {'ticker': 'TAP', 'sector': 'Consumer Staples'},
    {'ticker': 'ALLE', 'sector': 'Industrials'},
    {'ticker': 'HWM', 'sector': 'Industrials'},
    {'ticker': 'KSS', 'sector': 'Consumer Discretionary'},
    {'ticker': 'COHR', 'sector': 'Technology'},
    {'ticker': 'FFIV', 'sector': 'Technology'},
    {'ticker': 'NTAP', 'sector': 'Technology'},
    {'ticker': 'WHR', 'sector': 'Consumer Discretionary'},
    {'ticker': 'CTXS', 'sector': 'Technology'},
    {'ticker': 'PVH', 'sector': 'Consumer Discretionary'},
    {'ticker': 'NWL', 'sector': 'Consumer Discretionary'},
    {'ticker': 'GPS', 'sector': 'Consumer Discretionary'},
    {'ticker': 'FTV', 'sector': 'Industrials'},
    {'ticker': 'DXC', 'sector': 'Technology'},
    {'ticker': 'COTY', 'sector': 'Consumer Staples'},
    {'ticker': 'OMC', 'sector': 'Communication Services'},
    {'ticker': 'IPG', 'sector': 'Communication Services'},
    {'ticker': 'WYNN', 'sector': 'Consumer Discretionary'},
    {'ticker': 'HBI', 'sector': 'Consumer Discretionary'},
    {'ticker': 'UAA', 'sector': 'Consumer Discretionary'},
    {'ticker': 'AA', 'sector': 'Materials'},
    {'ticker': 'UA', 'sector': 'Consumer Discretionary'}
]

# Mock database file
MOCK_DB_PATH = os.path.join(os.path.dirname(__file__), "mock_stock_data.json")

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50  # Default neutral RSI
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.mean(gains[:period])
    avg_losses = np.mean(losses[:period])
    
    if avg_losses == 0:
        return 100
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ma(prices, period):
    """Calculate Moving Average"""
    if len(prices) < period:
        return np.mean(prices)
    return np.mean(prices[-period:])

def fetch_real_stock_data():
    """Fetch real historical data from Yahoo Finance"""
    data = {}
    total_stocks = len(SP500_STOCKS)
    successful_fetches = 0
    failed_fetches = 0
    
    logging.info(f"Starting to fetch data for {total_stocks} S&P 500 stocks...")
    
    for i, stock in enumerate(SP500_STOCKS):
        ticker = stock['ticker']
        logging.info(f"[{i+1}/{total_stocks}] Fetching data for {ticker}...")
        
        try:
            # Get the stock data from Yahoo Finance
            stock_obj = yf.Ticker(ticker)
            
            # Get 2 years of historical data
            hist = stock_obj.history(period="2y")
            
            if hist.empty:
                logging.warning(f"⚠️  No data available for {ticker}")
                failed_fetches += 1
                continue
            
            info = stock_obj.info
            sector = stock.get('sector', info.get('sector', 'Unknown'))
            market_cap = info.get('marketCap', random.randint(50000000000, 3000000000000))
            revenue_growth = info.get('revenueGrowth', random.uniform(-5, 20))
            
            data[ticker] = []
            prices = []
            
            for i, (date, row) in enumerate(hist.iterrows()):
                prices.append(row['Close'])
                
                # Calculate technical indicators
                rsi = calculate_rsi(prices)
                ma_20 = calculate_ma(prices, 20)
                ma_50 = calculate_ma(prices, 50)
                
                data[ticker].append({
                    'date': date.isoformat(),
                    'open': round(row['Open'], 2),
                    'high': round(row['High'], 2),
                    'low': round(row['Low'], 2),
                    'close': round(row['Close'], 2),
                    'volume': int(row['Volume']),
                    'rsi': round(rsi, 2),
                    'ma_20': round(ma_20, 2),
                    'ma_50': round(ma_50, 2),
                    'sector': sector,
                    'market_cap': market_cap,
                    'revenue_growth': round(revenue_growth * 100, 2) if isinstance(revenue_growth, (int, float)) else 5.0
                })
            
            successful_fetches += 1
            print(f"✅ Fetched {len(data[ticker])} days of data for {ticker}")
            
            # Add small delay to avoid rate limiting
            if (i + 1) % 50 == 0:
                print(f"⏸️  Taking a brief pause after {i + 1} stocks...")
                time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {str(e)}")
            # Create fallback mock data for failed stocks
            data[ticker] = generate_fallback_data(ticker, stock['sector'])
            failed_fetches += 1
    
    print(f"\n📊 Data Fetch Summary:")
    print(f"   Total stocks: {total_stocks}")
    print(f"   Successful: {successful_fetches}")
    print(f"   Failed: {failed_fetches}")
    print(f"   Success rate: {(successful_fetches/total_stocks)*100:.1f}%")
    
    return data

def generate_fallback_data(ticker, sector):
    """Generate fallback data when Yahoo Finance API fails"""
    fallback_data = []
    base_date = datetime.now() - timedelta(days=365*2)
    base_price = random.uniform(50, 500)
    
    for i in range(500):
        current_date = base_date + timedelta(days=i)
        daily_return = random.normalvariate(0.001, 0.02)
        base_price *= (1 + daily_return)
        
        open_price = base_price * random.uniform(0.99, 1.01)
        high_price = max(open_price, base_price) * random.uniform(1.00, 1.02)
        low_price = min(open_price, base_price) * random.uniform(0.98, 1.00)
        close_price = base_price
        
        rsi = 50 + random.normalvariate(0, 15)
        rsi = max(0, min(100, rsi))
        
        ma_20 = close_price * random.uniform(0.98, 1.02)
        ma_50 = close_price * random.uniform(0.96, 1.04)
        
        fallback_data.append({
            'date': current_date.isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': random.randint(1000000, 50000000),
            'rsi': round(rsi, 2),
            'ma_20': round(ma_20, 2),
            'ma_50': round(ma_50, 2),
            'sector': sector,
            'market_cap': random.randint(50000000000, 3000000000000),
            'revenue_growth': round(random.uniform(-5, 20), 2)
        })
    
    return fallback_data

def get_stock_data():
    """Get or fetch stock data"""
    if os.path.exists(MOCK_DB_PATH):
        with open(MOCK_DB_PATH, 'r') as f:
            return json.load(f)
    else:
        print("Fetching real stock data from Yahoo Finance...")
        data = fetch_real_stock_data()
        with open(MOCK_DB_PATH, 'w') as f:
            json.dump(data, f)
        print(f"✓ Saved {len(data)} stocks to database")
        return data

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/initialize")
async def initialize_data():
    """Initialize or reinitialize the database with real data"""
    try:
        logging.info("Initializing real stock database...")
        # Delete existing database to force fresh fetch
        if os.path.exists(MOCK_DB_PATH):
            os.remove(MOCK_DB_PATH)
            logging.info(f"Deleted existing database file: {MOCK_DB_PATH}")
        
        logging.info("Fetching real stock data from Yahoo Finance...")
        data = fetch_real_stock_data()
        with open(MOCK_DB_PATH, 'w') as f:
            json.dump(data, f)
        total_records = sum(len(stock_data) for stock_data in data.values())
        logging.info(f"Successfully saved {total_records} records to {MOCK_DB_PATH}")
        logging.info(f"Total records saved: {total_records}")
        
        return {
            "status": "created", 
            "message": f"Real stock database initialized with {total_records} records",
            "stocks": len(data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

@app.get("/api/stocks/search")
async def search_stocks(
    query: str = "",
    sector: str = "",
    min_market_cap: float = 0,
    max_rsi: float = 100,
    min_revenue_growth: float = 0,
    limit: int = 30
) -> List[Dict[str, Any]]:
    """Search stocks based on various criteria"""
    try:
        data = get_stock_data()
        
        # Get latest data for each stock
        stocks = []
        for ticker, stock_data in data.items():
            if not stock_data:
                continue
                
            latest = stock_data[-1]  # Get most recent data
            
            # Apply filters
            if query and query.lower() not in ticker.lower():
                continue
                
            if sector and latest['sector'] != sector:
                continue
            
            if latest['market_cap'] < min_market_cap:
                continue
            
            if latest['rsi'] > max_rsi:
                continue
            
            if latest['revenue_growth'] < min_revenue_growth:
                continue
            
            stocks.append({
                "ticker": ticker,
                "sector": latest['sector'],
                "market_cap": latest['market_cap'],
                "revenue_growth": latest['revenue_growth'],
                "current_price": latest['close'],
                "current_rsi": latest['rsi'],
                "ma_50": latest['ma_50']
            })
        
        # Sort by market cap and limit
        stocks.sort(key=lambda x: x['market_cap'], reverse=True)
        return stocks[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/api/stocks/{ticker}/news")
async def get_stock_news(ticker: str) -> List[Dict[str, Any]]:
    """Get recent news for a stock"""
    try:
        # Mock news data
        mock_news = [
            {
                "title": f"{ticker} shows strong momentum in Q4 earnings",
                "publisher": "Market Analysis",
                "link": "#",
                "timestamp": int(datetime.now().timestamp())
            },
            {
                "title": f"Analysts upgrade {ticker} price target amid sector rotation",
                "publisher": "Investment Weekly",
                "link": "#",
                "timestamp": int((datetime.now() - timedelta(days=1)).timestamp())
            },
            {
                "title": f"{ticker} announces new product line, shares rise",
                "publisher": "Tech News",
                "link": "#",
                "timestamp": int((datetime.now() - timedelta(days=2)).timestamp())
            }
        ]
        
        # Add some negative news occasionally for realism
        if random.random() < 0.3:
            mock_news.insert(0, {
                "title": f"{ticker} faces regulatory scrutiny over recent practices",
                "publisher": "Financial Times",
                "link": "#",
                "timestamp": int((datetime.now() - timedelta(hours=6)).timestamp())
            })
        
        return mock_news[:5]  # Return top 5 news items
        
    except Exception as e:
        return [{
            "title": f"{ticker} maintains steady performance in current market",
            "publisher": "Market Analysis",
            "link": "#",
            "timestamp": int(datetime.now().timestamp())
        }]

@app.get("/api/stocks/{ticker}/history")
async def get_stock_history(
    ticker: str,
    period: str = "1y"
) -> List[Dict[str, Any]]:
    logging.info(f"Fetching history for {ticker} with period {period}")
    if ticker in INMEM_STOCK_DATA and INMEM_STOCK_DATA[ticker]:
        stock_data = INMEM_STOCK_DATA[ticker][-252:] if len(INMEM_STOCK_DATA[ticker]) > 252 else INMEM_STOCK_DATA[ticker]
        return [
            {
                'date': item['date'],
                'open': item['open'],
                'high': item['high'],
                'low': item['low'],
                'close': item['close'],
                'volume': item['volume'],
                'rsi': item['rsi'],
                'ma_20': item['ma_20'],
                'ma_50': item['ma_50']
            }
            for item in stock_data
        ]
    # Prefer live fetch to avoid loading entire DB
    logging.info(f"Fetching history for {ticker} with period {period} from Yahoo Finance")
    try:
        stock_obj = yf.Ticker(ticker)
        hist = stock_obj.history(period=period)
        if hist is None or hist.empty:
            alt = yf.download(ticker, period=period, progress=False)
            hist = alt if alt is not None else pd.DataFrame()
        if not hist.empty:
            prices: List[float] = []
            live = []
            for date, row in hist.iterrows():
                close_val = row['Close'] if 'Close' in row else (row['Adj Close'] if 'Adj Close' in row else None)
                open_val = row['Open'] if 'Open' in row else close_val
                high_val = row['High'] if 'High' in row else close_val
                low_val = row['Low'] if 'Low' in row else close_val
                if close_val is None:
                    continue
                prices.append(float(close_val))
                rsi = calculate_rsi(prices)
                ma_20 = calculate_ma(prices, 20)
                ma_50 = calculate_ma(prices, 50)
                vol_value = row['Volume'] if 'Volume' in row else 0
                if pd.isna(vol_value):
                    vol_value = 0
                live.append({
                    'date': date.isoformat(),
                    'open': round(float(open_val), 2),
                    'high': round(float(high_val), 2),
                    'low': round(float(low_val), 2),
                    'close': round(float(close_val), 2),
                    'volume': int(vol_value),
                    'rsi': round(rsi, 2),
                    'ma_20': round(ma_20, 2),
                    'ma_50': round(ma_50, 2)
                })
            return live
    except Exception:
        pass

    # Fall back to local database if available
    try:
        data = get_stock_data()
        if ticker in data and data[ticker]:
            stock_data = data[ticker][-252:] if len(data[ticker]) > 252 else data[ticker]
            return [
                {
                    'date': item['date'],
                    'open': item['open'],
                    'high': item['high'],
                    'low': item['low'],
                    'close': item['close'],
                    'volume': item['volume'],
                    'rsi': item['rsi'],
                    'ma_20': item['ma_20'],
                    'ma_50': item['ma_50']
                }
                for item in stock_data
            ]
    except Exception:
        pass

    # Last resort: generated fallback
    fb = generate_fallback_data(ticker, 'Unknown')
    trimmed = fb[-252:] if len(fb) > 252 else fb
    return [
        {
            'date': it['date'],
            'open': it['open'],
            'high': it['high'],
            'low': it['low'],
            'close': it['close'],
            'volume': it['volume'],
            'rsi': it['rsi'],
            'ma_20': it['ma_20'],
            'ma_50': it['ma_50'],
        }
        for it in trimmed
    ]

# SpoonAI Agent Endpoints

@app.post("/api/agents/build-portfolio")
async def build_portfolio_with_agents():
    """Build portfolio using SpoonAI multi-agent system"""
    try:
        if not trading_orchestrator:
            raise HTTPException(status_code=503, detail="Trading orchestrator agent not initialized")
        
        # Initialize trading state
        initial_state = TradingState(
            scout_candidates=[],
            evaluated_stocks=[],
            sentiment_scores={},
            final_rankings=[],
            portfolio=[],
            current_step="initialized",
            iteration_count=0,
            target_portfolio_size=20,
            natural_query=""
        )
        
        print("🚀 Starting multi-agent portfolio building process...")
        
        # Run the orchestrator
        result = await trading_orchestrator.run(json.dumps(initial_state))
        
        # Parse the result
        final_state = json.loads(result) if isinstance(result, str) else result
        
        portfolio = final_state.get("portfolio", [])
        
        print(f"✅ Portfolio building completed with {len(portfolio)} stocks")
        
        return {
            "status": "success",
            "portfolio": portfolio,
            "total_stocks": len(portfolio),
            "iterations": final_state.get("iteration_count", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio building failed: {str(e)}")

@app.post("/api/agents/scout")
async def scout_stocks(request: Dict[str, Any]):
    """Scout for stock candidates using ScoutAgent with optional natural language query"""
    try:
        if not scout_agent:
            raise HTTPException(status_code=503, detail="Scout agent not initialized")
        
        from agents import ScoutState
        
        query = request.get("query", "")
        stock_count = int(request.get("stock_count", 20))
        explicit_criteria = request.get("criteria", {})
        
        scout_state = ScoutState(
            candidates=[],
            search_criteria=explicit_criteria,
            stock_count=stock_count
        )
        # Pass natural query for LLM interpretation inside agent
        state_payload = {**scout_state, "natural_query": query}  # type: ignore
        
        result = await scout_agent.run(json.dumps(state_payload))
        scout_result = json.loads(result) if isinstance(result, str) else result
        
        return {
            "status": "success",
            "candidates": scout_result.get("candidates", []),
            "criteria": scout_result.get("search_criteria", {}),
            "total_found": len(scout_result.get("candidates", [])),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scouting failed: {str(e)}")

@app.post("/api/agents/evaluate")
async def evaluate_stocks(stocks: List[Dict[str, Any]]):
    """Evaluate stocks using EvaluationAgent"""
    try:
        if not evaluation_agent:
            raise HTTPException(status_code=503, detail="Evaluation agent not initialized")
        
        from agents import EvaluationState
        
        eval_state = EvaluationState(
            stocks_to_evaluate=stocks,
            evaluation_results=[],
            fundamental_metrics={},
            technical_metrics={}
        )
        
        result = await evaluation_agent.run(json.dumps(eval_state))
        eval_result = json.loads(result) if isinstance(result, str) else result
        
        passed_stocks = eval_result.get("passed_stocks", [])
        
        return {
            "status": "success",
            "evaluation_results": eval_result.get("evaluation_results", []),
            "passed_stocks": passed_stocks,
            "total_passed": len(passed_stocks),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/api/agents/sentiment")
async def analyze_sentiment(request: Request):
    """Analyze sentiment for stocks using SentimentAgent"""
    try:
        if not sentiment_agent:
            raise HTTPException(status_code=503, detail="Sentiment agent not initialized")
        
        # Parse JSON array from request body
        tickers = await request.json()
        if not isinstance(tickers, list):
            raise HTTPException(status_code=400, detail="Request body must be a JSON array of tickers")
        
        from agents import SentimentState
        
        sentiment_state = SentimentState(
            stocks_to_analyze=tickers,
            news_data={},
            social_data={},
            sentiment_scores={}
        )
        
        result = await sentiment_agent.run(json.dumps(sentiment_state))
        sentiment_result = json.loads(result) if isinstance(result, str) else result
        
        return {
            "status": "success",
            "sentiment_scores": sentiment_result.get("sentiment_scores", {}),
            "total_analyzed": len(tickers),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/api/agents/rank")
async def rank_stocks(request: Dict[str, Any]):
    """Rank stocks using RankingAgent"""
    try:
        if not ranking_agent:
            raise HTTPException(status_code=503, detail="Ranking agent not initialized")
        
        from agents import RankingState
        
        evaluated_stocks = request.get("evaluated_stocks", [])
        sentiment_scores = request.get("sentiment_scores", {})
        weights = request.get("weights", {"fundamental": 5, "technical": 3, "sentiment": 2})
        
        ranking_state = RankingState(
            evaluated_stocks=evaluated_stocks,
            sentiment_scores=sentiment_scores,
            final_scores=[],
            weights=weights
        )
        
        result = await ranking_agent.run(json.dumps(ranking_state))
        ranking_result = json.loads(result) if isinstance(result, str) else result
        
        final_scores = ranking_result.get("final_scores", [])
        
        return {
            "status": "success",
            "final_rankings": final_scores,
            "total_ranked": len(final_scores),
            "top_10": final_scores[:10],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")

@app.post("/api/agents/analyze")
async def analyze_portfolio(request: Dict[str, Any]):
    try:
        portfolio = request.get("portfolio", [])
        backtest = request.get("backtest_results", [])
        total_positions = len(portfolio)
        avg_weight = round(100 / total_positions, 1) if total_positions else 0
        avg_score = round(sum([p.get("score", 0) for p in portfolio]) / total_positions, 2) if total_positions else 0
        avg_alpha = 0
        if backtest:
            avg_alpha = round(sum([r.get("alpha", 0) for r in backtest]) / len(backtest), 2)
        top_positions = ", ".join([p.get("ticker") for p in portfolio[:5]])
        summary_lines = []
        summary_lines.append(f"Total positions: {total_positions}")
        if total_positions:
            summary_lines.append(f"Equal-weight target: {avg_weight}%")
            summary_lines.append(f"Average score: {avg_score}")
            summary_lines.append(f"Top positions: {top_positions}")
        if backtest:
            summary_lines.append(f"Average alpha vs SPY: {avg_alpha}%")
        sector_counts: Dict[str, int] = {}
        for p in portfolio:
            sector = p.get("sector", "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if sector_counts:
            dominant = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            sector_line = ", ".join([f"{s}: {c}" for s, c in dominant])
            summary_lines.append(f"Sector distribution: {sector_line}")
        summary = "\n".join(summary_lines)
        return {"status": "success", "summary": summary, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/memory/append")
async def memory_append(request: Dict[str, Any]):
    try:
        thread_id = request.get("thread_id", "default")
        role = request.get("role", "assistant")
        content = request.get("content", "")
        meta = request.get("meta", {})
        _ensure_thread(thread_id)
        if Message:
            msg = Message(id=str(uuid.uuid4()), role=role, content=content, metadata=meta)
        else:
            msg = {"id": str(uuid.uuid4()), "role": role, "content": content, "metadata": meta}
        MEMORY_THREADS[thread_id].append(msg)
        return {"status": "success", "count": len(MEMORY_THREADS[thread_id])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory append failed: {str(e)}")

@app.post("/api/memory/get")
async def memory_get(request: Dict[str, Any]):
    try:
        thread_id = request.get("thread_id", "default")
        _ensure_thread(thread_id)
        msgs = MEMORY_THREADS[thread_id]
        out = []
        for m in msgs:
            if Message and hasattr(m, "id"):
                out.append({"id": m.id, "role": m.role, "content": m.content, "metadata": getattr(m, "metadata", {})})
            else:
                out.append(m)
        return {"status": "success", "messages": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory get failed: {str(e)}")

@app.post("/api/memory/clear")
async def memory_clear(request: Dict[str, Any]):
    try:
        thread_id = request.get("thread_id", "default")
        MEMORY_THREADS[thread_id] = []
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory clear failed: {str(e)}")

@app.get("/api/agents/status")
async def get_agents_status():
    """Get status of all SpoonAI agents"""
    return {
        "trading_orchestrator": trading_orchestrator is not None,
        "scout_agent": scout_agent is not None,
        "evaluation_agent": evaluation_agent is not None,
        "sentiment_agent": sentiment_agent is not None,
        "ranking_agent": ranking_agent is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")