# Just Buy It API

FastAPI backend for the Just Buy It trading platform.

## Features

- **Data Lake**: DuckDB integration with 10 years of S&P 500 data
- **Real-time Data**: yfinance integration for live market data
- **AI Agents**: Backend services for Scout, Evaluation, Portfolio, and Backtest agents
- **REST API**: Full REST API for frontend integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the development server:
```bash
python main.py
```

The API will be available at `http://localhost:8001`

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/initialize` - Initialize database with S&P 500 data
- `GET /api/stocks/search` - Search stocks with filters
- `GET /api/stocks/{ticker}/news` - Get stock news
- `GET /api/stocks/{ticker}/history` - Get stock price history

## Database Schema

The DuckDB database contains:
- Stock price data (OHLCV)
- Technical indicators (RSI, Moving Averages)
- Fundamental data (market cap, revenue growth)
- Sector information