"""
SpoonAI Agent Implementation for Quantitative Trading Platform
"""

from typing import TypedDict, List, Dict, Any, Optional
import asyncio
import json
import random
from datetime import datetime
import httpx
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uuid

try:
    from spoon_ai.agents.toolcall import ToolCallAgent
    from spoon_ai.chat import ChatBot
    from spoon_ai.tools import ToolManager
    from spoon_ai.tools.base import BaseTool
    SPOON_AVAILABLE = True
    print("✅ SpoonAI components loaded successfully")
except Exception:
    ToolCallAgent = object  # type: ignore
    ChatBot = object  # type: ignore
    ToolManager = object  # type: ignore
    BaseTool = object  # type: ignore
    SPOON_AVAILABLE = False
    print("⚠️ SpoonAI components not available")



class MockStateGraph:
    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
        self.finish_points = []
        self.compiled = False
    
    def add_node(self, name, node):
        self.nodes[name] = node
    
    def add_edge(self, from_node, to_node, condition=None):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append((to_node, condition))
        print(f"📊 Added edge: {from_node} -> {to_node} with condition: {condition}")
    
    def set_entry_point(self, node_name):
        self.entry_point = node_name
    
    def set_finish_point(self, node_name):
        self.finish_points.append(node_name)
    
    def compile(self):
        self.compiled = True
        return self
    
    async def run(self, initial_state):
        """Execute the graph with the given initial state"""
        print(f"🔄 Starting graph execution with entry point: {self.entry_point}")
        print(f"📋 Available nodes: {list(self.nodes.keys())}")
        print(f"📋 Available edges: {self.edges}")
        print(f"🏁 Finish points: {self.finish_points}")
        
        current_state = json.loads(initial_state) if isinstance(initial_state, str) else initial_state
        current_node = self.entry_point
        step_count = 0
        executed_nodes = []  # Track executed nodes to detect cycles
        
        while current_node and current_node not in self.finish_points and step_count < 20:  # Safety limit
            step_count += 1
            print(f"📍 Step {step_count}: Executing node: {current_node}")
            print(f"📝 Current state before execution: {current_state}")
            print(f"🔍 Executed nodes so far: {executed_nodes}")
            
            if current_node in executed_nodes:
                print(f"⚠️ Node {current_node} already executed, possible cycle detected")
                print(f"🔄 Breaking cycle by stopping execution")
                break
                
            if current_node not in self.nodes:
                print(f"❌ Node {current_node} not found in nodes: {list(self.nodes.keys())}")
                break
            
            node = self.nodes[current_node]
            print(f"🎯 Found node: {node.name if hasattr(node, 'name') else 'unnamed'}")
            
            # Execute node
            if hasattr(node, 'func'):
                print(f"🚀 Executing function for node: {current_node}")
                print(f"📝 Node function: {node.func.__name__ if hasattr(node.func, '__name__') else 'unnamed'}")
                result = await node.func(current_state)
                print(f"📊 Node {current_node} result: {result}")
            elif hasattr(node, 'condition_func'):
                # Condition nodes don't execute, they just determine next node
                print(f"🎯 Condition node {current_node} - skipping execution")
                result = {}
            else:
                print(f"⚠️ Node {current_node} has no function")
                result = {}
            
            # Update state - preserve existing fields and only update returned fields
            for key, value in result.items():
                if key != "current_step" or value != current_state.get("current_step"):  # Only update current_step if it's different
                    current_state[key] = value
            print(f"📝 Updated state after {current_node}: {current_state}")
            
            # Track executed node
            executed_nodes.append(current_node)
            
            # Determine next node
            if current_node in self.edges:
                edges = self.edges[current_node]
                print(f"🔗 Node {current_node} has edges: {[(edge[0], type(edge[1]).__name__ if edge[1] else 'None') for edge in edges]}")
                if len(edges) == 1:
                    current_node, condition = edges[0]
                    print(f"➡️ Moving to next node: {current_node}")
                else:
                    # Handle conditional edges with condition function
                    print(f"🔍 Processing conditional edges for {current_node}")
                    
                    # Get the condition result (target node name)
                    if hasattr(edges[0][1], 'condition_func'):  # First condition is a ConditionNode
                        target_node = edges[0][1].condition_func(current_state)
                    else:  # Direct condition function
                        target_node = edges[0][1](current_state)
                    
                    print(f"🎯 Condition result: {target_node}")
                    
                    # Find the edge that matches the target
                    for next_node, condition in edges:
                        if next_node == target_node:
                            current_node = next_node
                            print(f"➡️ Moving to conditional node: {current_node}")
                            break
                    else:
                        print(f"❌ No edge found for target: {target_node}")
                        break
            else:
                print(f"🏁 No edges from {current_node}, finishing")
                break
        
        print(f"✅ Graph execution completed after {step_count} steps")
        return json.dumps(current_state)


class MockGraphAgent:
    def __init__(self, name: str, graph, preserve_state: bool = False):
        self.name = name
        self.graph = graph
        self.preserve_state = preserve_state
        self.memory = {}
    
    async def run(self, request: str) -> str:
        """Run the agent with the given request"""
        try:
            result = await self.graph.run(request)
            return result
        except Exception as e:
            print(f"Error running agent {self.name}: {e}")
            return json.dumps({"error": str(e)})


class MockRunnableNode:
    def __init__(self, name: str, func):
        self.name = name
        self.func = func


class MockConditionNode:
    def __init__(self, name: str, condition_func):
        self.name = name
        self.condition_func = condition_func


class MockToolNode:
    def __init__(self, name: str, tools):
        self.name = name
        self.tools = tools


StateGraph = MockStateGraph
GraphAgent = MockGraphAgent
RunnableNode = MockRunnableNode
ConditionNode = MockConditionNode
ToolNode = MockToolNode
END = "END"

if SPOON_AVAILABLE:
    class SearchStocksTool(BaseTool):
        name: str = "search_stocks"
        description: str = "Search stocks using criteria"
        parameters: dict = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "min_market_cap": {"type": "number"},
                "max_rsi": {"type": "number"},
                "min_revenue_growth": {"type": "number"},
            },
            "required": ["limit"]
        }
        async def execute(self, limit: int, min_market_cap: float = 0, max_rsi: float = 100, min_revenue_growth: float = 0) -> List[Dict[str, Any]]:
            params = {
                "limit": limit,
                "min_market_cap": min_market_cap,
                "max_rsi": max_rsi,
                "min_revenue_growth": min_revenue_growth,
            }
            async with httpx.AsyncClient() as client:
                r = await client.get("http://localhost:8001/api/stocks/search", params=params)
                if r.status_code == 200:
                    return r.json()
            return []

    class EvaluateStocksTool(BaseTool):
        name: str = "evaluate_stocks"
        description: str = "Evaluate stocks using fundamentals and technicals"
        parameters: dict = {
            "type": "object",
            "properties": {
                "stocks": {"type": "array"}
            },
            "required": ["stocks"]
        }
        async def execute(self, stocks: List[Dict[str, Any]]) -> Dict[str, Any]:
            evaluation_results = []
            for stock in stocks:
                ticker = stock["ticker"]
                detailed = await fetch_stock_data_from_api(ticker)
                fundamental = random.uniform(5, 8)
                if stock.get("revenue_growth", 0) > 10:
                    fundamental += 1.5
                if stock.get("market_cap", 0) > 10000000000:
                    fundamental += 1
                technical = 5
                if detailed["rsi"] < 30:
                    technical += 2
                elif detailed["rsi"] > 70:
                    technical -= 2
                if detailed["current_price"] > detailed["ma_20"]:
                    technical += 1
                if detailed["ma_20"] > detailed["ma_50"]:
                    technical += 2
                fundamental = max(0, min(10, fundamental))
                technical = max(0, min(10, technical))
                evaluation_passed = (
                    fundamental >= 6.0 and technical >= 5.0 and stock.get("market_cap", 0) > 10000000000
                )
                evaluation_results.append({
                    "ticker": ticker,
                    "sector": stock.get("sector", "Unknown"),
                    "current_price": detailed["current_price"],
                    "fundamental_score": round(fundamental, 2),
                    "technical_score": round(technical, 2),
                    "rsi": detailed["rsi"],
                    "market_cap": stock.get("market_cap", 0),
                    "revenue_growth": stock.get("revenue_growth", 0),
                    "evaluation_passed": evaluation_passed
                })
            passed = [s for s in evaluation_results if s["evaluation_passed"]]
            return {"evaluation_results": evaluation_results, "passed_stocks": passed}

    class SentimentTool(BaseTool):
        name: str = "sentiment"
        description: str = "Analyze sentiment for tickers"
        parameters: dict = {
            "type": "object",
            "properties": {
                "tickers": {"type": "array"}
            },
            "required": ["tickers"]
        }
        async def execute(self, tickers: List[str]) -> Dict[str, float]:
            scores = {}
            for t in tickers:
                scores[t] = round(random.uniform(4, 9), 2)
            return scores

    class RankTool(BaseTool):
        name: str = "rank"
        description: str = "Rank evaluated stocks with sentiment"
        parameters: dict = {
            "type": "object",
            "properties": {
                "evaluated_stocks": {"type": "array"},
                "sentiment_scores": {"type": "object"},
                "weights": {"type": "object"},
            },
            "required": ["evaluated_stocks", "sentiment_scores"]
        }
        async def execute(self, evaluated_stocks: List[Dict[str, Any]], sentiment_scores: Dict[str, float], weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
            if weights is None:
                weights = {"fundamental": 5, "technical": 3, "sentiment": 2}
            final_scores = []
            for stock in evaluated_stocks:
                fs = stock.get("fundamental_score", 0)
                ts = stock.get("technical_score", 0)
                ss = sentiment_scores.get(stock["ticker"], 5.0)
                weighted = (
                    fs * weights["fundamental"] + ts * weights["technical"] + ss * weights["sentiment"]
                ) / sum(weights.values())
                final_scores.append({
                    "ticker": stock["ticker"],
                    "sector": stock.get("sector", "Unknown"),
                    "current_price": stock.get("current_price", 0),
                    "fundamental_score": fs,
                    "technical_score": ts,
                    "sentiment_score": ss,
                    "weighted_score": round(weighted, 2),
                    "market_cap": stock.get("market_cap", 0),
                    "revenue_growth": stock.get("revenue_growth", 0)
                })
            final_scores.sort(key=lambda x: x["weighted_score"], reverse=True)
            return final_scores

    class SpoonScoutAgent(ToolCallAgent):
        def __init__(self):
            super().__init__(
                llm=ChatBot(llm_provider="gemini", model_name="gpt-4.1"),
                available_tools=ToolManager([SearchStocksTool()])
            )
            print("✅ SpoonScoutAgent initialized with Gemini LLM")
        
        async def run(self, request: str) -> str:  # type: ignore
            state = json.loads(request) if isinstance(request, str) else request
            criteria = state.get("search_criteria", {})
            limit = state.get("stock_count", criteria.get("limit", 20))
            params = {
                "limit": limit,
                "min_market_cap": criteria.get("min_market_cap", 0),
                "max_rsi": criteria.get("max_rsi", 100),
                "min_revenue_growth": criteria.get("min_revenue_growth", 0)
            }
            data = await self.available_tools.tools[0].execute(**params)
            return json.dumps({"candidates": data, "search_criteria": params})

    class SpoonEvaluationAgent(ToolCallAgent):
        def __init__(self):
            super().__init__(
                llm=ChatBot(llm_provider="gemini", model_name="gpt-4.1"),
                available_tools=ToolManager([EvaluateStocksTool()])
            )
        async def run(self, request: str) -> str:  # type: ignore
            state = json.loads(request) if isinstance(request, str) else request
            stocks = state.get("stocks_to_evaluate", [])
            result = await self.available_tools.tools[0].execute(stocks=stocks)
            return json.dumps(result)

    class SpoonSentimentAgent(ToolCallAgent):
        def __init__(self):
            super().__init__(
                llm=ChatBot(llm_provider="gemini", model_name="gpt-4.1"),
                available_tools=ToolManager([SentimentTool()])
            )
            print("✅ SpoonSentimentAgent initialized with Gemini LLM")
        async def run(self, request: str) -> str:  # type: ignore
            state = json.loads(request) if isinstance(request, str) else request
            tickers = state.get("stocks_to_analyze", [])
            scores = await self.available_tools.tools[0].execute(tickers=tickers)
            return json.dumps({"sentiment_scores": scores})

    class SpoonRankingAgent(ToolCallAgent):
        def __init__(self):
            super().__init__(
                llm=ChatBot(llm_provider="gemini", model_name="gpt-4.1"),
                available_tools=ToolManager([RankTool()])
            )
            print("✅ SpoonRankingAgent initialized with Gemini LLM")
        async def run(self, request: str) -> str:  # type: ignore
            state = json.loads(request) if isinstance(request, str) else request
            evaluated = state.get("evaluated_stocks", [])
            sentiments = state.get("sentiment_scores", {})
            weights = state.get("weights", {"fundamental": 5, "technical": 3, "sentiment": 2})
            ranked = await self.available_tools.tools[0].execute(evaluated_stocks=evaluated, sentiment_scores=sentiments, weights=weights)
            return json.dumps({"final_scores": ranked})


# State definitions
class TradingState(TypedDict):
    """Main state for trading workflow"""
    scout_candidates: List[Dict[str, Any]]
    evaluated_stocks: List[Dict[str, Any]]
    sentiment_scores: Dict[str, float]
    final_rankings: List[Dict[str, Any]]
    portfolio: List[Dict[str, Any]]
    current_step: str
    iteration_count: int
    target_portfolio_size: int


class ScoutState(TypedDict):
    """State for ScoutAgent"""
    candidates: List[Dict[str, Any]]
    search_criteria: Dict[str, Any]
    stock_count: int


class EvaluationState(TypedDict):
    """State for EvaluationAgent"""
    stocks_to_evaluate: List[Dict[str, Any]]
    evaluation_results: List[Dict[str, Any]]
    fundamental_metrics: Dict[str, Any]
    technical_metrics: Dict[str, Any]


class SentimentState(TypedDict):
    """State for SentimentAgent"""
    stocks_to_analyze: List[str]
    news_data: Dict[str, List[Dict[str, Any]]]
    social_data: Dict[str, List[Dict[str, Any]]]
    sentiment_scores: Dict[str, float]


class RankingState(TypedDict):
    """State for RankingAgent"""
    evaluated_stocks: List[Dict[str, Any]]
    sentiment_scores: Dict[str, float]
    final_scores: List[Dict[str, Any]]
    weights: Dict[str, float]


# Utility functions
async def fetch_stock_data_from_api(ticker: str) -> Dict[str, Any]:
    """Fetch stock data from the trading platform API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://localhost:8001/api/stocks/{ticker}/history")
            if response.status_code == 200:
                data = response.json()
                if data:
                    latest = data[-1]
                    return {
                        "ticker": ticker,
                        "current_price": latest["close"],
                        "rsi": latest["rsi"],
                        "ma_20": latest["ma_20"],
                        "ma_50": latest["ma_50"],
                        "volume": latest["volume"]
                    }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    
    # Fallback data
    return {
        "ticker": ticker,
        "current_price": random.uniform(50, 500),
        "rsi": random.uniform(30, 70),
        "ma_20": random.uniform(45, 495),
        "ma_50": random.uniform(40, 490),
        "volume": random.randint(1000000, 50000000)
    }


async def search_stocks_from_api(criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Search stocks using the trading platform API"""
    print(f"Searching stocks with criteria: {criteria}")
    
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "limit": criteria.get("limit", 30),
                "min_market_cap": criteria.get("min_market_cap", 0),
                "max_rsi": criteria.get("max_rsi", 100),
                "min_revenue_growth": criteria.get("min_revenue_growth", 0)
            }
            
            response = await client.get("http://localhost:8001/api/stocks/search", params=params)
            if response.status_code == 200:
                data = response.json()
                print(f"API returned {len(data)} stocks")
                return data
    except Exception as e:
        print(f"Error searching stocks via API: {e}")
    
    # Fallback: return mock S&P 500 stocks
    print("Using fallback mock data")
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
                     'XOM', 'JPM', 'V', 'PG', 'HD', 'MA', 'CVX', 'LLY', 'ABBV', 'PFE']
    
    candidates = []
    limit = min(criteria.get("limit", 20), len(sp500_tickers))
    selected_tickers = random.sample(sp500_tickers, limit)
    
    for ticker in selected_tickers:
        candidates.append({
            "ticker": ticker,
            "sector": random.choice(['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary']),
            "market_cap": random.randint(50000000000, 3000000000000),
            "revenue_growth": random.uniform(-5, 20),
            "current_price": random.uniform(50, 500),
            "current_rsi": random.uniform(30, 70)
        })
    
    print(f"Generated {len(candidates)} mock candidates")
    return candidates


# Node implementations
async def scout_node(state: ScoutState) -> Dict[str, Any]:
    """Scout node to find stock candidates"""
    print(f"🔍 ScoutAgent: Searching for {state['stock_count']} stock candidates...")
    
    criteria = state.get("search_criteria", {})
    criteria["limit"] = state["stock_count"]
    
    candidates = await search_stocks_from_api(criteria)
    
    print(f"✅ ScoutAgent: Found {len(candidates)} candidates")
    
    return {
        "candidates": candidates,
        "search_criteria": criteria
    }


async def evaluation_node(state: EvaluationState) -> Dict[str, Any]:
    """Evaluation node to analyze fundamentals and technical data"""
    print(f"📊 EvaluationAgent: Evaluating {len(state['stocks_to_evaluate'])} stocks...")
    
    evaluation_results = []
    
    for stock in state["stocks_to_evaluate"]:
        ticker = stock["ticker"]
        print(f"  Evaluating {ticker}...")
        
        # Fetch detailed data
        detailed_data = await fetch_stock_data_from_api(ticker)
        print(f"    Data: RSI={detailed_data['rsi']}, Price=${detailed_data['current_price']}, MA20=${detailed_data['ma_20']}, MA50=${detailed_data['ma_50']}")
        
        # Fundamental analysis (reasonable for normal operation)
        fundamental_score = random.uniform(5, 8)  # Reasonable base range
        if stock.get("revenue_growth", 0) > 10:  # Good growth
            fundamental_score += 1.5
        if stock.get("market_cap", 0) > 100000000000:  # Large cap
            fundamental_score += 1
        
        # Technical analysis (reasonable for normal operation)
        technical_score = 5  # Reasonable base score
        if detailed_data["rsi"] < 30:  # Oversold
            technical_score += 2
        elif detailed_data["rsi"] > 70:  # Overbought
            technical_score -= 2
        
        if detailed_data["current_price"] > detailed_data["ma_20"]:  # Above MA20
            technical_score += 1
        
        if detailed_data["ma_20"] > detailed_data["ma_50"]:  # Golden cross
            technical_score += 2
        
        # Normalize scores
        fundamental_score = max(0, min(10, fundamental_score))
        technical_score = max(0, min(10, technical_score))
        
        # Reasonable evaluation criteria for normal operation
        market_cap = stock.get("market_cap", 0)
        print(f"    DEBUG: Checking criteria - Fundamental={fundamental_score:.2f} >= 6.0? {fundamental_score >= 6.0}")
        print(f"    DEBUG: Checking criteria - Technical={technical_score:.2f} >= 5.0? {technical_score >= 5.0}")
        print(f"    DEBUG: Checking criteria - Market Cap={market_cap:,.0f} > 10B? {market_cap > 10000000000}")
        
        evaluation_passed = (
            fundamental_score >= 6.0 and  # Good fundamental score
            technical_score >= 5.0 and   # Decent technical score
            market_cap > 10000000000  # Large cap (>10B)
        )
        print(f"    Scores: Fundamental={fundamental_score:.2f}, Technical={technical_score:.2f}, Market Cap={market_cap:,.0f}, Passed={evaluation_passed}")
        
        evaluation_results.append({
            "ticker": ticker,
            "sector": stock.get("sector", "Unknown"),
            "current_price": detailed_data["current_price"],
            "fundamental_score": round(fundamental_score, 2),
            "technical_score": round(technical_score, 2),
            "rsi": detailed_data["rsi"],
            "market_cap": stock.get("market_cap", 0),
            "revenue_growth": stock.get("revenue_growth", 0),
            "evaluation_passed": evaluation_passed
        })
    
    passed_stocks = [stock for stock in evaluation_results if stock["evaluation_passed"]]
    print(f"✅ EvaluationAgent: {len(passed_stocks)} stocks passed evaluation out of {len(evaluation_results)}")
    print(f"📊 Sample evaluation results: {evaluation_results[:3] if evaluation_results else 'None'}")
    
    return {
        "evaluation_results": evaluation_results,
        "passed_stocks": passed_stocks
    }


async def sentiment_node(state: SentimentState) -> Dict[str, Any]:
    """Sentiment node to analyze news and social media sentiment"""
    print(f"💭 SentimentAgent: Analyzing sentiment for {len(state['stocks_to_analyze'])} stocks...")
    
    sentiment_scores = {}
    
    for ticker in state["stocks_to_analyze"]:
        try:
            print(f"🔍 Analyzing sentiment for {ticker}...")
            
            # For now, use random sentiment scores to test the pipeline
            # In a real implementation, this would fetch news and analyze sentiment
            sentiment_score = random.uniform(4, 9)  # Random sentiment between 4-9
            sentiment_scores[ticker] = round(sentiment_score, 2)
            print(f"💯 Sentiment score for {ticker}: {sentiment_scores[ticker]:.2f}")
                    
        except Exception as e:
            print(f"❌ Error analyzing sentiment for {ticker}: {e}")
            sentiment_scores[ticker] = random.uniform(4, 7)
            print(f"⚠️ Using fallback sentiment for {ticker}: {sentiment_scores[ticker]:.2f}")
    
    print(f"✅ SentimentAgent: Completed sentiment analysis for {len(sentiment_scores)} stocks")
    print(f"📊 Sentiment scores: {sentiment_scores}")
    
    return {
        "sentiment_scores": sentiment_scores
    }


async def ranking_node(state: RankingState) -> Dict[str, Any]:
    """Ranking node to score stocks with weighted algorithm"""
    print(f"🏆 RankingAgent: Ranking {len(state['evaluated_stocks'])} stocks...")
    print(f"📊 Available sentiment scores: {len(state.get('sentiment_scores', {}))} stocks")
    
    weights = state.get("weights", {
        "fundamental": 5,
        "technical": 3,
        "sentiment": 2
    })
    
    final_scores = []
    
    for stock in state["evaluated_stocks"]:
        ticker = stock["ticker"]
        
        # Get sentiment score
        sentiment_score = state["sentiment_scores"].get(ticker, 5.0)
        
        # Calculate weighted score
        fundamental_score = stock.get("fundamental_score", 0)
        technical_score = stock.get("technical_score", 0)
        
        weighted_score = (
            (fundamental_score * weights["fundamental"]) +
            (technical_score * weights["technical"]) +
            (sentiment_score * weights["sentiment"])
        ) / sum(weights.values())
        
        final_scores.append({
            "ticker": ticker,
            "sector": stock.get("sector", "Unknown"),
            "current_price": stock.get("current_price", 0),
            "fundamental_score": fundamental_score,
            "technical_score": technical_score,
            "sentiment_score": sentiment_score,
            "weighted_score": round(weighted_score, 2),
            "market_cap": stock.get("market_cap", 0),
            "revenue_growth": stock.get("revenue_growth", 0)
        })
        
        print(f"📈 {ticker}: Fundamental={fundamental_score}, Technical={technical_score}, Sentiment={sentiment_score:.2f}, Weighted={weighted_score:.2f}")
    
    # Sort by weighted score (descending)
    final_scores.sort(key=lambda x: x["weighted_score"], reverse=True)
    
    print(f"✅ RankingAgent: Completed ranking of {len(final_scores)} stocks")
    print(f"🏆 Top 5 ranked stocks: {[stock['ticker'] for stock in final_scores[:5]]}")
    
    return {
        "final_scores": final_scores
    }


# Graph construction functions
def create_scout_graph() -> StateGraph:
    """Create ScoutAgent graph"""
    graph = StateGraph(ScoutState)
    
    # Add scout node
    scout_runnable = RunnableNode("scout", scout_node)
    graph.add_node("scout", scout_runnable)
    
    # Set entry point - don't set finish point to same node
    graph.set_entry_point("scout")
    # The graph will finish when there are no more edges
    
    return graph.compile()


def create_evaluation_graph() -> StateGraph:
    """Create EvaluationAgent graph"""
    graph = StateGraph(EvaluationState)
    
    # Add evaluation node
    eval_runnable = RunnableNode("evaluate", evaluation_node)
    graph.add_node("evaluate", eval_runnable)
    
    # Set entry point only - don't set finish point to same node
    graph.set_entry_point("evaluate")
    # Let it finish naturally after execution
    
    return graph.compile()


def create_sentiment_graph() -> StateGraph:
    """Create SentimentAgent graph"""
    graph = StateGraph(SentimentState)
    
    # Add sentiment node
    sentiment_runnable = RunnableNode("sentiment", sentiment_node)
    graph.add_node("sentiment", sentiment_runnable)
    
    # Set entry point only
    graph.set_entry_point("sentiment")
    # Let it finish naturally after execution
    
    return graph.compile()


def create_ranking_graph() -> StateGraph:
    """Create RankingAgent graph"""
    graph = StateGraph(RankingState)
    
    # Add ranking node
    ranking_runnable = RunnableNode("rank", ranking_node)
    graph.add_node("rank", ranking_runnable)
    
    # Set entry point only
    graph.set_entry_point("rank")
    # Let it finish naturally after execution
    
    return graph.compile()


# Main orchestration graph
def create_trading_orchestrator_graph() -> StateGraph:
    """Create main trading orchestrator graph"""
    graph = StateGraph(TradingState)
    
    # Create sub-graph agents
    scout_agent = GraphAgent("scout", create_scout_graph())
    eval_agent = GraphAgent("evaluation", create_evaluation_graph())
    sentiment_agent = GraphAgent("sentiment", create_sentiment_graph())
    ranking_agent = GraphAgent("ranking", create_ranking_graph())
    
    # Define orchestrator nodes
    async def initialize_portfolio(state: TradingState) -> Dict[str, Any]:
        """Initialize portfolio building process"""
        print("🚀 Initializing portfolio building process...")
        return {
            "current_step": "initialized",
            "iteration_count": 0,
            "target_portfolio_size": 20
        }
    
    async def scout_stocks(state: TradingState) -> Dict[str, Any]:
        """Scout for stock candidates"""
        print(f"🔍 Scouting iteration {state['iteration_count'] + 1}...")
        
        # Determine how many stocks to scout
        remaining_needed = state["target_portfolio_size"] - len(state["evaluated_stocks"])
        scout_count = max(20, remaining_needed * 2)  # Scout more than needed
        
        scout_state = ScoutState(
            candidates=[],
            search_criteria={"limit": scout_count, "min_market_cap": 10000000000},  # Large cap focus
            stock_count=scout_count
        )
        
        result = await scout_agent.run(json.dumps(scout_state))
        scout_result = json.loads(result) if isinstance(result, str) else result
        
        return {
            "scout_candidates": scout_result.get("candidates", []),
            "current_step": "scouting_completed"
        }
    
    async def evaluate_stocks(state: TradingState) -> Dict[str, Any]:
        """Evaluate scouted stocks"""
        print(f"📊 Evaluating {len(state['scout_candidates'])} candidates...")
        
        eval_state = EvaluationState(
            stocks_to_evaluate=state["scout_candidates"],
            evaluation_results=[],
            fundamental_metrics={},
            technical_metrics={}
        )
        
        print(f"🔄 Running evaluation agent...")
        result = await eval_agent.run(json.dumps(eval_state))
        print(f"📋 Evaluation result: {result}")
        
        eval_result = json.loads(result) if isinstance(result, str) else result
        
        passed_stocks = eval_result.get("passed_stocks", [])
        print(f"✅ {len(passed_stocks)} stocks passed evaluation")
        print(f"📋 First few passed stocks: {passed_stocks[:2] if passed_stocks else 'None'}")
        
        # Combine with existing evaluated stocks
        all_evaluated = state["evaluated_stocks"] + passed_stocks
        
        return {
            "evaluated_stocks": all_evaluated,
            "current_step": "evaluation_completed"
        }
    
    async def analyze_sentiment(state: TradingState) -> Dict[str, Any]:
        """Analyze sentiment for evaluated stocks"""
        tickers = [stock["ticker"] for stock in state["evaluated_stocks"]]
        print(f"💭 Analyzing sentiment for {len(tickers)} stocks...")
        
        sentiment_state = SentimentState(
            stocks_to_analyze=tickers,
            news_data={},
            social_data={},
            sentiment_scores={}
        )
        
        print(f"🎯 Calling sentiment agent with state: {sentiment_state}")
        result = await sentiment_agent.run(json.dumps(sentiment_state))
        sentiment_result = json.loads(result) if isinstance(result, str) else result
        
        print(f"📊 Sentiment analysis result: {sentiment_result}")
        sentiment_scores = sentiment_result.get("sentiment_scores", {})
        print(f"💯 Sentiment scores: {sentiment_scores}")
        
        return {
            "sentiment_scores": sentiment_scores,
            "current_step": "sentiment_completed"
        }
    
    async def rank_stocks(state: TradingState) -> Dict[str, Any]:
        """Rank evaluated stocks"""
        print(f"🏆 Ranking {len(state['evaluated_stocks'])} stocks...")
        print(f"📊 Available sentiment scores: {state.get('sentiment_scores', {})}")
        
        ranking_state = RankingState(
            evaluated_stocks=state["evaluated_stocks"],
            sentiment_scores=state["sentiment_scores"],
            final_scores=[],
            weights={"fundamental": 5, "technical": 3, "sentiment": 2}
        )
        
        print(f"🎯 Calling ranking agent with state: {len(ranking_state['evaluated_stocks'])} stocks")
        result = await ranking_agent.run(json.dumps(ranking_state))
        ranking_result = json.loads(result) if isinstance(result, str) else result
        
        print(f"📊 Ranking result: {ranking_result}")
        final_rankings = ranking_result.get("final_scores", [])
        print(f"💯 Final rankings count: {len(final_rankings)}")
        
        return {
            "final_rankings": final_rankings,
            "current_step": "ranking_completed"
        }
    
    async def build_portfolio(state: TradingState) -> Dict[str, Any]:
        """Build final portfolio"""
        print("📈 Building final portfolio...")
        
        # Select top stocks
        top_stocks = state["final_rankings"][:state["target_portfolio_size"]]
        
        # Calculate portfolio weights (equal weight for simplicity)
        portfolio_weight = 1.0 / len(top_stocks) if top_stocks else 0
        
        portfolio = []
        for stock in top_stocks:
            portfolio.append({
                "ticker": stock["ticker"],
                "weight": portfolio_weight,
                "score": stock["weighted_score"],
                "sector": stock.get("sector", "Unknown"),
                "current_price": stock.get("current_price", 0)
            })
        
        return {
            "portfolio": portfolio,
            "current_step": "portfolio_completed"
        }
    
    async def check_portfolio_completion(state: TradingState) -> Dict[str, Any]:
        """Check if portfolio is complete or needs more stocks"""
        current_size = len(state["portfolio"])
        target_size = state["target_portfolio_size"]
        
        if current_size >= target_size:
            print(f"✅ Portfolio complete with {current_size} stocks")
            return {"current_step": "completed"}
        else:
            print(f"📈 Need {target_size - current_size} more stocks")
            return {
                "current_step": "needs_more_stocks",
                "iteration_count": state["iteration_count"] + 1
            }
    
    # Add nodes to graph
    graph.add_node("initialize", RunnableNode("initialize", initialize_portfolio))
    graph.add_node("scout", RunnableNode("scout", scout_stocks))
    graph.add_node("evaluate", RunnableNode("evaluate", evaluate_stocks))
    graph.add_node("sentiment", RunnableNode("sentiment", analyze_sentiment))
    graph.add_node("rank", RunnableNode("rank", rank_stocks))
    graph.add_node("build_portfolio", RunnableNode("build_portfolio", build_portfolio))
    graph.add_node("check_completion", RunnableNode("check_completion", check_portfolio_completion))
    
    # Add conditional node for iteration logic
    def should_continue_scouting(state: TradingState) -> str:
        if state["current_step"] == "needs_more_stocks" and state["iteration_count"] < 5:
            return "scout"  # Return the actual node name to continue to
        else:
            return "END"  # Return the actual end node name
    
    continue_node = ConditionNode("should_continue", should_continue_scouting)
    graph.add_node("should_continue", continue_node)
    
    # Define edges
    graph.add_edge("initialize", "scout")
    graph.add_edge("scout", "evaluate")
    graph.add_edge("evaluate", "sentiment")
    graph.add_edge("sentiment", "rank")
    graph.add_edge("rank", "build_portfolio")
    graph.add_edge("build_portfolio", "check_completion")
    graph.add_edge("check_completion", "should_continue")
    
    # Conditional edges - use the condition function to determine which edge to take
    graph.add_edge("should_continue", "scout", should_continue_scouting)
    graph.add_edge("should_continue", END, should_continue_scouting)
    
    # Set entry point and finish points
    graph.set_entry_point("initialize")
    graph.set_finish_point("END")  # Add END as a finish point
    
    return graph.compile()


# Agent factory functions
def create_trading_orchestrator_agent() -> GraphAgent:
    """Create the main trading orchestrator agent"""
    return GraphAgent(
        name="trading_orchestrator",
        graph=create_trading_orchestrator_graph(),
        preserve_state=True
    )


def create_scout_agent() -> GraphAgent:
    if SPOON_AVAILABLE:
        return SpoonScoutAgent()
    return GraphAgent(
        name="scout_agent",
        graph=create_scout_graph(),
        preserve_state=False
    )


def create_evaluation_agent() -> GraphAgent:
    if SPOON_AVAILABLE:
        return SpoonEvaluationAgent()
    return GraphAgent(
        name="evaluation_agent",
        graph=create_evaluation_graph(),
        preserve_state=False
    )


def create_sentiment_agent() -> GraphAgent:
    if SPOON_AVAILABLE:
        return SpoonSentimentAgent()
    return GraphAgent(
        name="sentiment_agent",
        graph=create_sentiment_graph(),
        preserve_state=False
    )


def create_ranking_agent() -> GraphAgent:
    if SPOON_AVAILABLE:
        return SpoonRankingAgent()
    return GraphAgent(
        name="ranking_agent",
        graph=create_ranking_graph(),
        preserve_state=False
    )


# Export all agents
__all__ = [
    "create_trading_orchestrator_agent",
    "create_scout_agent",
    "create_evaluation_agent",
    "create_sentiment_agent",
    "create_ranking_agent",
    "TradingState",
    "ScoutState",
    "EvaluationState",
    "SentimentState",
    "RankingState"
]