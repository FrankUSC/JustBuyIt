# Just Buy It

Autonomous AI quantitative trading platform with a React frontend and FastAPI backend orchestrated by SpoonAI Graph agents. Scouts candidates, evaluates fundamentals/technicals, analyzes sentiment, ranks by weighted scores, and builds a 20‑stock portfolio.

## Project Structure
- `src/` React frontend (Vite + TypeScript)
- `api/` FastAPI backend with multi‑agent orchestration
- `doc/` Design prompts and notes

## Prerequisites
- `Python >= 3.10` and `pip`
- `Node.js >= 18`

## Backend Setup (FastAPI)
1. `cd api`
2. (optional) create venv
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv && .venv\\Scripts\\activate`
3. Install deps: `pip install -r requirements.txt`
4. Start server: `python main.py`

Backend runs at `http://localhost:8001`.

### Key Agent Endpoints
- `POST /api/agents/build-portfolio` → Runs full multi‑agent pipeline, returns final portfolio
- `POST /api/agents/scout?stock_count=20&min_market_cap=10000000000` → Returns candidates
- `POST /api/agents/evaluate` (body: JSON array of stocks) → Returns `passed_stocks`
- `POST /api/agents/sentiment` (body: JSON array of tickers) → Returns `sentiment_scores`
- `POST /api/agents/rank` (body: `{ evaluated_stocks, sentiment_scores, weights }`) → Returns `final_rankings`
- `GET /api/agents/status` → Agent initialization status

Example:
```bash
# Build portfolio
curl -X POST http://localhost:8001/api/agents/build-portfolio

# Evaluate a stock
curl -X POST http://localhost:8001/api/agents/evaluate \
  -H "Content-Type: application/json" \
  -d '[{"ticker":"AAPL","sector":"Technology","market_cap":2800000000000,"revenue_growth":8}]'
```

## Frontend Setup (Vite + React)
1. In project root: `npm install`
2. Start dev server: `npm run dev`
3. Open `http://localhost:5173`

## Workflow Summary
- Scout → Evaluate → Sentiment → Rank → Portfolio
- Ranking weights: fundamentals `5`, technical `3`, sentiment `2`
- Iterative orchestration continues scouting until 20 stocks pass evaluation and ranking, then builds an equal‑weight portfolio.

## Notes
- CORS is enabled for local dev (`5173`, `5174`, `3000`)
- Data sources: Yahoo Finance (via `yfinance`) with fallbacks
- Sentiment currently uses a deterministic pipeline placeholder; plug in real news/social sources as needed
