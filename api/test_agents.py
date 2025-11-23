#!/usr/bin/env python3
"""
Test script for SpoonAI agents
"""

import asyncio
import json
import sys
from agents import (
    create_scout_agent,
    create_evaluation_agent,
    create_sentiment_agent,
    create_ranking_agent,
    create_trading_orchestrator_agent,
    ScoutState,
    EvaluationState,
    SentimentState,
    RankingState,
    TradingState,
)

async def test_scout_agent():
    print("🧪 Testing ScoutAgent...")
    try:
        scout_agent = create_scout_agent()
        print("✅ ScoutAgent created successfully")
        scout_state = ScoutState(
            candidates=[],
            search_criteria={"limit": 5, "min_market_cap": 10000000000},
            stock_count=5,
            natural_query="Find 5 large cap technology stocks with revenue growth",
        )
        print(f"📝 Initial state: {json.dumps(scout_state, indent=2)}")
        result = await scout_agent.run(json.dumps(scout_state))
        print(f"📊 Result: {result}")
        result_data = json.loads(result)
        print(f"✅ Found {len(result_data.get('candidates', []))} candidates")
        if result_data.get('candidates'):
            print("First candidate:", result_data['candidates'][0])
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_evaluation_agent():
    print("🧪 Testing EvaluationAgent...")
    try:
        evaluation_agent = create_evaluation_agent()
        print("✅ EvaluationAgent created successfully")
        stocks = [
            {"ticker": "AAPL", "sector": "Technology", "market_cap": 2500000000000, "revenue_growth": 12.5},
            {"ticker": "MSFT", "sector": "Technology", "market_cap": 2300000000000, "revenue_growth": 10.1},
            {"ticker": "GOOGL", "sector": "Technology", "market_cap": 1800000000000, "revenue_growth": 8.7},
            {"ticker": "AMZN", "sector": "Consumer Discretionary", "market_cap": 1700000000000, "revenue_growth": 15.3},
            {"ticker": "NVDA", "sector": "Technology", "market_cap": 1200000000000, "revenue_growth": 20.0},
            {"ticker": "META", "sector": "Technology", "market_cap": 900000000000, "revenue_growth": 13.2},
            {"ticker": "BRK-B", "sector": "Financials", "market_cap": 800000000000, "revenue_growth": 6.5},
            {"ticker": "UNH", "sector": "Healthcare", "market_cap": 450000000000, "revenue_growth": 9.0},
            {"ticker": "JNJ", "sector": "Healthcare", "market_cap": 380000000000, "revenue_growth": 5.4},
            {"ticker": "XOM", "sector": "Energy", "market_cap": 400000000000, "revenue_growth": 7.8},
            {"ticker": "JPM", "sector": "Financials", "market_cap": 500000000000, "revenue_growth": 8.1},
            {"ticker": "V", "sector": "Financials", "market_cap": 550000000000, "revenue_growth": 11.0},
            {"ticker": "PG", "sector": "Consumer Staples", "market_cap": 340000000000, "revenue_growth": 4.2},
            {"ticker": "HD", "sector": "Consumer Discretionary", "market_cap": 300000000000, "revenue_growth": 6.0},
            {"ticker": "MA", "sector": "Financials", "market_cap": 450000000000, "revenue_growth": 10.8},
            {"ticker": "CVX", "sector": "Energy", "market_cap": 300000000000, "revenue_growth": 6.9},
            {"ticker": "LLY", "sector": "Healthcare", "market_cap": 600000000000, "revenue_growth": 18.5},
            {"ticker": "ABBV", "sector": "Healthcare", "market_cap": 300000000000, "revenue_growth": 7.1},
            {"ticker": "PFE", "sector": "Healthcare", "market_cap": 200000000000, "revenue_growth": 3.8},
            {"ticker": "KO", "sector": "Consumer Staples", "market_cap": 270000000000, "revenue_growth": 5.0},
            {"ticker": "PEP", "sector": "Consumer Staples", "market_cap": 250000000000, "revenue_growth": 5.6},
            {"ticker": "WMT", "sector": "Consumer Staples", "market_cap": 470000000000, "revenue_growth": 4.7},
            {"ticker": "MRK", "sector": "Healthcare", "market_cap": 280000000000, "revenue_growth": 6.3},
            {"ticker": "BAC", "sector": "Financials", "market_cap": 260000000000, "revenue_growth": 4.9},
        ]
        eval_state = EvaluationState(
            stocks_to_evaluate=stocks,
            evaluation_results=[],
            fundamental_metrics={},
            technical_metrics={},
        )
        print(f"📝 Initial state: {json.dumps(eval_state, indent=2)}")
        result = await evaluation_agent.run(json.dumps(eval_state))
        print(f"📊 Result: {result}")
        data = json.loads(result)
        print(f"✅ Evaluated {len(data.get('evaluation_results', []))} stocks")
        passed = data.get('passed_stocks', [])
        print(f"✅ Passed {len(passed)} stocks")
        if passed:
            print("First passed:", passed[0])
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_sentiment_agent():
    print("🧪 Testing SentimentAgent...")
    try:
        sentiment_agent = create_sentiment_agent()
        print("✅ SentimentAgent created successfully")
        sentiment_state = SentimentState(
            stocks_to_analyze=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            news_data={},
            social_data={},
            sentiment_scores={},
        )
        print(f"📝 Initial state: {json.dumps(sentiment_state, indent=2)}")
        result = await sentiment_agent.run(json.dumps(sentiment_state))
        print(f"📊 Result: {result}")
        data = json.loads(result)
        scores = data.get('sentiment_scores', {})
        print(f"✅ Scores for {len(scores)} tickers")
        items = list(scores.items())
        if items:
            print("Sample:", items[:3])
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_ranking_agent():
    print("🧪 Testing RankingAgent...")
    try:
        ranking_agent = create_ranking_agent()
        print("✅ RankingAgent created successfully")
        evaluated_stocks = [
            {"ticker": "AAPL", "sector": "Technology", "current_price": 190.0, "fundamental_score": 7.5, "technical_score": 6.2, "market_cap": 2500000000000, "revenue_growth": 12.5},
            {"ticker": "MSFT", "sector": "Technology", "current_price": 380.0, "fundamental_score": 7.2, "technical_score": 6.0, "market_cap": 2300000000000, "revenue_growth": 10.1},
            {"ticker": "GOOGL", "sector": "Technology", "current_price": 140.0, "fundamental_score": 6.8, "technical_score": 5.5, "market_cap": 1800000000000, "revenue_growth": 8.7},
            {"ticker": "AMZN", "sector": "Consumer Discretionary", "current_price": 150.0, "fundamental_score": 7.0, "technical_score": 5.8, "market_cap": 1700000000000, "revenue_growth": 15.3},
            {"ticker": "NVDA", "sector": "Technology", "current_price": 480.0, "fundamental_score": 8.0, "technical_score": 7.0, "market_cap": 1200000000000, "revenue_growth": 20.0},
        ]
        sentiment_scores = {"AAPL": 7.8, "MSFT": 7.5, "GOOGL": 7.0, "AMZN": 6.5, "NVDA": 8.5}
        ranking_state = RankingState(
            evaluated_stocks=evaluated_stocks,
            sentiment_scores=sentiment_scores,
            final_scores=[],
            weights={"fundamental": 5, "technical": 3, "sentiment": 2},
        )
        print(f"📝 Initial state: {json.dumps(ranking_state, indent=2)}")
        result = await ranking_agent.run(json.dumps(ranking_state))
        print(f"📊 Result: {result}")
        data = json.loads(result)
        final_scores = data.get('final_scores', [])
        print(f"✅ Ranked {len(final_scores)} stocks")
        if final_scores:
            print("Top 3:", final_scores[:3])
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_orchestrator_agent():
    print("🧪 Testing Trading Orchestrator Agent...")
    try:
        orchestrator = create_trading_orchestrator_agent()
        print("✅ Orchestrator created successfully")
        initial_state = TradingState(
            scout_candidates=[],
            evaluated_stocks=[],
            sentiment_scores={},
            final_rankings=[],
            portfolio=[],
            current_step="initialized",
            iteration_count=0,
            target_portfolio_size=10,
            natural_query="Find large cap technology stocks with low RSI",
        )
        print(f"📝 Initial state: {json.dumps(initial_state, indent=2)}")
        result = await orchestrator.run(json.dumps(initial_state))
        print(f"📊 Result: {result}")
        data = json.loads(result)
        print(f"✅ Portfolio size: {len(data.get('portfolio', []))}")
        if data.get('portfolio'):
            print("Sample portfolio item:", data['portfolio'][0])
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target == "scout":
        await test_scout_agent()
    elif target == "eval":
        await test_evaluation_agent()
    elif target == "sentiment":
        await test_sentiment_agent()
    elif target == "ranking":
        await test_ranking_agent()
    elif target == "orchestrator":
        await test_orchestrator_agent()
    else:
        await test_scout_agent()
        await test_evaluation_agent()
        await test_sentiment_agent()
        await test_ranking_agent()

if __name__ == "__main__":
    asyncio.run(main())