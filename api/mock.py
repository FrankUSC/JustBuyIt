from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any
import asyncio
import random

app = FastAPI(title="Pocket Hedge Fund API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mock data for S&P 500 stocks
TOP_STOCKS = [
    {'ticker': 'AAPL', 'sector': 'Technology', 'market_cap': 3000000000000, 'revenue_growth': 8.2},
    {'ticker': 'MSFT', 'sector': 'Technology', 'market_cap': 2800000000000, 'revenue_growth': 12.1},
    {'ticker': 'GOOGL', 'sector': 'Technology', 'market_cap': 1700000000000, 'revenue_growth': 15.3},
    {'ticker': 'AMZN', 'sector': 'Consumer Discretionary', 'market_cap': 1600000000000, 'revenue_growth': 9.8},
    {'ticker': 'NVDA', 'sector': 'Technology', 'market_cap': 1200000000000, 'revenue_growth': 22.4},
    {'ticker': 'TSLA', 'sector': 'Consumer Discretionary', 'market_cap': 800000000000, 'revenue_growth': 18.7},
    {'ticker': 'META', 'sector': 'Technology', 'market_cap': 900000000000, 'revenue_growth': 11.2},
    {'ticker': 'BRK-B', 'sector': 'Financials', 'market_cap': 700000000000, 'revenue_growth': 5.1},
    {'ticker': 'UNH', 'sector': 'Healthcare', 'market_cap': 500000000000, 'revenue_growth': 13.4},
    {'ticker': 'JNJ', 'sector': 'Healthcare', 'market_cap': 450000000000, 'revenue_growth': 6.8},
    {'ticker': 'XOM', 'sector': 'Energy', 'market_cap': 400000000000, 'revenue_growth': -2.3},
    {'ticker': 'JPM', 'sector': 'Financials', 'market_cap': 500000000000, 'revenue_growth': 7.9},
    {'ticker': 'V', 'sector': 'Financials', 'market_cap': 480000000000, 'revenue_growth': 14.2},
    {'ticker': 'PG', 'sector': 'Consumer Staples', 'market_cap': 380000000000, 'revenue_growth': 4.1},
    {'ticker': 'HD', 'sector': 'Consumer Discretionary', 'market_cap': 350000000000, 'revenue_growth': 6.7},
    {'ticker': 'MA', 'sector': 'Financials', 'market_cap': 360000000000, 'revenue_growth': 16.3},
    {'ticker': 'CVX', 'sector': 'Energy', 'market_cap': 320000000000, 'revenue_growth': -1.8},
    {'ticker': 'LLY', 'sector': 'Healthcare', 'market_cap': 300000000000, 'revenue_growth': 19.8},
    {'ticker': 'ABBV', 'sector': 'Healthcare', 'market_cap': 280000000000, 'revenue_growth': 8.9},
    {'ticker': 'PFE', 'sector': 'Healthcare', 'market_cap': 200000000000, 'revenue_growth': -3.2},
    {'ticker': 'KO', 'sector': 'Consumer Staples', 'market_cap': 270000000000, 'revenue_growth': 5.4},
    {'ticker': 'PEP', 'sector': 'Consumer Staples', 'market_cap': 240000000000, 'revenue_growth': 7.8},
    {'ticker': 'WMT', 'sector': 'Consumer Staples', 'market_cap': 600000000000, 'revenue_growth': 4.9},
    {'ticker': 'MRK', 'sector': 'Healthcare', 'market_cap': 260000000000, 'revenue_growth': 9.1},
    {'ticker': 'BAC', 'sector': 'Financials', 'market_cap': 280000000000, 'revenue_growth': 11.7},
    {'ticker': 'ADBE', 'sector': 'Technology', 'market_cap': 220000000000, 'revenue_growth': 12.4},
    {'ticker': 'NFLX', 'sector': 'Communication Services', 'market_cap': 180000000000, 'revenue_growth': 6.7},
    {'ticker': 'CRM', 'sector': 'Technology', 'market_cap': 200000000000, 'revenue_growth': 18.9},
    {'ticker': 'ACN', 'sector': 'Information Technology', 'market_cap': 190000000000, 'revenue_growth': 8.3},
    {'ticker': 'TMO', 'sector': 'Healthcare', 'market_cap': 210000000000, 'revenue_growth': 14.6}
]

# Mock database file
MOCK_DB_PATH = "mock_stock_data.json"

def generate_mock_stock_data():
    """Generate mock historical data for stocks"""
    data = {}
    base_date = datetime.now() - timedelta(days=365*2)  # 2 years of data
    
    for stock in TOP_STOCKS:
        ticker = stock['ticker']
        data[ticker] = []
        base_price = random.uniform(50, 500)  # Random base price
        
        for i in range(500):  # 500 days of data
            current_date = base_date + timedelta(days=i)
            
            # Generate realistic price movements
            daily_return = random.normalvariate(0.001, 0.02)  # Small positive drift with volatility
            base_price *= (1 + daily_return)
            
            # Generate OHLC data
            open_price = base_price * random.uniform(0.99, 1.01)
            high_price = max(open_price, base_price) * random.uniform(1.00, 1.02)
            low_price = min(open_price, base_price) * random.uniform(0.98, 1.00)
            close_price = base_price
            
            # Calculate technical indicators
            rsi = 50 + random.normalvariate(0, 15)  # RSI around 50
            rsi = max(0, min(100, rsi))  # Clamp to 0-100
            
            ma_20 = close_price * random.uniform(0.98, 1.02)
            ma_50 = close_price * random.uniform(0.96, 1.04)
            
            data[ticker].append({
                'date': current_date.isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.randint(1000000, 50000000),
                'rsi': round(rsi, 2),
                'ma_20': round(ma_20, 2),
                'ma_50': round(ma_50, 2),
                'sector': stock['sector'],
                'market_cap': stock['market_cap'],
                'revenue_growth': stock['revenue_growth']
            })
    
    return data

def get_mock_data():
    """Get or generate mock stock data"""
    if os.path.exists(MOCK_DB_PATH):
        with open(MOCK_DB_PATH, 'r') as f:
            return json.load(f)
    else:
        data = generate_mock_stock_data()
        with open(MOCK_DB_PATH, 'w') as f:
            json.dump(data, f)
        return data

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/initialize")
async def initialize_data():
    """Initialize or reinitialize the mock database"""
    try:
        data = generate_mock_stock_data()
        with open(MOCK_DB_PATH, 'w') as f:
            json.dump(data, f)
        
        total_records = sum(len(stock_data) for stock_data in data.values())
        
        return {
            "status": "created", 
            "message": f"Mock database initialized with {total_records} records",
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
        data = get_mock_data()
        
        # Get latest data for each stock
        stocks = []
        for ticker, stock_data in data.items():
            if not stock_data:
                continue
                
            latest = stock_data[-1]  # Get most recent data
            
            # Apply filters
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
    """Get historical price data for a stock"""
    try:
        data = get_mock_data()
        
        if ticker not in data or not data[ticker]:
            raise HTTPException(status_code=404, detail=f"Stock {ticker} not found")
        
        # Return last 252 trading days (about 1 year)
        stock_data = data[ticker][-252:] if len(data[ticker]) > 252 else data[ticker]
        
        history = []
        for item in stock_data:
            history.append({
                "date": item['date'],
                "open": item['open'],
                "high": item['high'],
                "low": item['low'],
                "close": item['close'],
                "volume": item['volume'],
                "rsi": item['rsi'],
                "ma_20": item['ma_20'],
                "ma_50": item['ma_50']
            })
        
        return history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)