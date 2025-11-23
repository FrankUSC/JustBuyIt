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
import os
from dotenv import load_dotenv
import logging
import sys

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)
logger.propagate = False

try:
    from spoon_ai.agents.toolcall import ToolCallAgent
    from spoon_ai.chat import ChatBot
    from spoon_ai.tools import ToolManager
    from spoon_ai.tools.base import BaseTool
    from spoon_ai.schema import Message
    SPOON_AVAILABLE = True
    logger.info("✅ SpoonAI components loaded successfully")
except Exception:
    ToolCallAgent = object  # type: ignore
    ChatBot = object  # type: ignore
    ToolManager = object  # type: ignore
    BaseTool = object  # type: ignore
    Message = dict  # type: ignore
    SPOON_AVAILABLE = False
    logger.warning("⚠️ SpoonAI components not available")



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
        logger.debug(f"📊 Added edge: {from_node} -> {to_node} with condition: {condition}")
    
    def set_entry_point(self, node_name):
        self.entry_point = node_name
    
    def set_finish_point(self, node_name):
        self.finish_points.append(node_name)
    
    def compile(self):
        self.compiled = True
        return self
    
    async def run(self, initial_state):
        """Execute the graph with the given initial state"""
        logger.debug(f"🔄 Starting graph execution with entry point: {self.entry_point}")
        logger.debug(f"📋 Available nodes: {list(self.nodes.keys())}")
        logger.debug(f"📋 Available edges: {self.edges}")
        logger.debug(f"🏁 Finish points: {self.finish_points}")
        
        current_state = json.loads(initial_state) if isinstance(initial_state, str) else initial_state
        current_node = self.entry_point
        step_count = 0
        executed_nodes = []  # Track executed nodes to detect cycles
        
        while current_node and current_node not in self.finish_points and step_count < 20:  # Safety limit
            step_count += 1
            logger.debug(f"📍 Step {step_count}: Executing node: {current_node}")
            logger.debug(f"📝 Current state before execution: {current_state}")
            logger.debug(f"🔍 Executed nodes so far: {executed_nodes}")
            
            if current_node in executed_nodes:
                logger.warning(f"⚠️ Node {current_node} already executed, possible cycle detected")
                logger.warning(f"🔄 Breaking cycle by stopping execution")
                break
                
            if current_node not in self.nodes:
                logger.error(f"❌ Node {current_node} not found in nodes: {list(self.nodes.keys())}")
                break
            
            node = self.nodes[current_node]
            logger.debug(f"🎯 Found node: {node.name if hasattr(node, 'name') else 'unnamed'}")
            
            # Execute node
            if hasattr(node, 'func'):
                logger.debug(f"🚀 Executing function for node: {current_node}")
                logger.debug(f"📝 Node function: {node.func.__name__ if hasattr(node.func, '__name__') else 'unnamed'}")
                result = await node.func(current_state)
                logger.debug(f"📊 Node {current_node} result: {result}")
            elif hasattr(node, 'condition_func'):
                # Condition nodes don't execute, they just determine next node
                logger.debug(f"🎯 Condition node {current_node} - skipping execution")
                result = {}
            else:
                logger.warning(f"⚠️ Node {current_node} has no function")
                result = {}
            
            # Update state - preserve existing fields and only update returned fields
            for key, value in result.items():
                if key != "current_step" or value != current_state.get("current_step"):  # Only update current_step if it's different
                    current_state[key] = value
            logger.debug(f"📝 Updated state after {current_node}: {current_state}")
            
            # Track executed node
            executed_nodes.append(current_node)
            
            # Determine next node
            if current_node in self.edges:
                edges = self.edges[current_node]
                logger.debug(f"🔗 Node {current_node} has edges: {[(edge[0], type(edge[1]).__name__ if edge[1] else 'None') for edge in edges]}")
                if len(edges) == 1:
                    current_node, condition = edges[0]
                    logger.debug(f"➡️ Moving to next node: {current_node}")
                else:
                    # Handle conditional edges with condition function
                    logger.debug(f"🔍 Processing conditional edges for {current_node}")
                    
                    # Get the condition result (target node name)
                    if hasattr(edges[0][1], 'condition_func'):  # First condition is a ConditionNode
                        target_node = edges[0][1].condition_func(current_state)
                    else:  # Direct condition function
                        target_node = edges[0][1](current_state)
                    
                    logger.debug(f"🎯 Condition result: {target_node}")
                    
                    # Find the edge that matches the target
                    for next_node, condition in edges:
                        if next_node == target_node:
                            current_node = next_node
                            logger.debug(f"➡️ Moving to conditional node: {current_node}")
                            break
                    else:
                        logger.error(f"❌ No edge found for target: {target_node}")
                        break
            else:
                logger.info(f"🏁 No edges from {current_node}, finishing")
                break
        
        logger.info(f"✅ Graph execution completed after {step_count} steps")
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
            logger.error(f"Error running agent {self.name}: {e}")
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
                "sector": {"type": "string"},
            },
            "required": ["limit"]
        }
        async def execute(self, limit: int, min_market_cap: float = 0, max_rsi: float = 100, min_revenue_growth: float = 0, sector: str = "") -> List[Dict[str, Any]]:
            params = {
                "limit": limit,
                "min_market_cap": min_market_cap,
                "max_rsi": max_rsi,
                "min_revenue_growth": min_revenue_growth,
                "sector": sector,
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
            logger.info(f"Evaluating {len(stocks)} stocks")
            fundamental_scores = await llm_evaluate_fundamentals(stocks)
            for stock in stocks:
                ticker = stock["ticker"]
                detailed = await fetch_stock_data_from_api(ticker)
                fundamental = fundamental_scores.get(ticker, 50)
                technical = 50
                if detailed["rsi"] < 30:
                    technical += 20
                elif detailed["rsi"] > 70:
                    technical -= 20
                if detailed["current_price"] > detailed["ma_20"]:
                    technical += 10
                if detailed["ma_20"] > detailed["ma_50"]:
                    technical += 20
                fundamental = max(0, min(100, fundamental))
                technical = max(0, min(100, technical))
                evaluation_passed = (
                    fundamental >= 60.0 and technical >= 50.0
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

    async def llm_evaluate_fundamentals(stocks: List[Dict[str, Any]]) -> Dict[str, int]:
        try:
            payload = []
            for s in stocks:
                d = {
                    "ticker": s.get("ticker"),
                    "sector": s.get("sector"),
                    "market_cap": s.get("market_cap"),
                    "revenue_growth": s.get("revenue_growth"),
                    "current_price": s.get("current_price"),
                    "rsi": s.get("rsi") or s.get("current_rsi"),
                    "ma_20": s.get("ma_20"),
                    "ma_50": s.get("ma_50"),
                    "volume": s.get("volume"),
                }
                payload.append(d)
            prompt = (
                "Given the following detailed stock data, research and evaluate the fundamentals of each stock. "
                "Return ONLY a JSON object mapping ticker to an integer score from 1 to 100 representing fundamental quality. "
                "Use sector context, market_cap, revenue_growth, margins if implied, stability, balance sheet, cache flow, profitability and quality. "
                "Stocks: " + json.dumps(payload)
            )
            logger.debug(f"🔍 Fundamentals prompt length: {len(prompt)}")
            if SPOON_AVAILABLE:
                try:
                    llm = ChatBot(llm_provider="gemini", model_name=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-001"))
                    result_text = None
                    if hasattr(llm, "ask"):
                        try:
                            msgs = [Message(role="user", content=prompt)]  
                        except Exception:
                            msgs = [{"role": "user", "content": prompt}]
                        rt = llm.ask(msgs)
                        result_text = await rt if asyncio.iscoroutine(rt) else rt
                    elif hasattr(llm, "chat"):
                        try:
                            sysm = Message(role="system", content="Return JSON only.")  
                            usrm = Message(role="user", content=prompt)  
                            payload_msgs = [sysm, usrm]
                        except Exception:
                            payload_msgs = [
                                {"role": "system", "content": "Return JSON only."},
                                {"role": "user", "content": prompt},
                            ]
                        res = llm.chat(payload_msgs)
                        res = await res if asyncio.iscoroutine(res) else res
                        result_text = res["content"] if isinstance(res, dict) and "content" in res else str(res)
                    else:
                        logger.warning("❌ LLM interface not available")
                        result_text = None
                    if not result_text:
                        logger.warning("❌ No LLM result for fundamentals evaluation")
                        raise RuntimeError("no_llm_result")
                    import re
                    m = re.search(r"\{[\s\S]*\}", result_text)
                    text = m.group(0) if m else result_text
                    data = json.loads(text)
                    out: Dict[str, int] = {}
                    if isinstance(data, dict):
                        for k, v in data.items():
                            try:
                                score = int(round(float(v)))
                            except Exception:
                                score = 50
                            score = max(1, min(100, score))
                            out[str(k)] = score
                    logger.debug(f"📊 Fundamentals scores received: {out}")
                    return out
                except Exception as e:
                    logger.error(f"❌ LLM fundamentals evaluation error: {e}")
                    e.print_stack()
                    return {}
            else:
                logger.warning("❌ No LLM result for fundamentals evaluation, fallback to basic scoring")
            out: Dict[str, int] = {}
            for s in payload:
                base = 50.0
                rg = s.get("revenue_growth") or 0.0
                mc = s.get("market_cap") or 0.0
                rsi = s.get("rsi") or 50.0
                base += (float(rg) - 5.0) * 2.0
                if mc > 100_000_000_000:
                    base += 10
                elif mc > 10_000_000_000:
                    base += 5
                if rsi < 30:
                    base += 2
                score = int(max(1, min(100, round(base))))
                out[str(s.get("ticker"))] = score
            logger.debug(f"📊 Fundamentals scores fallback: {out}")
            return out
        except Exception as e:
            logger.error(f"❌ Fundamentals evaluation failed: {e}")
            return {}

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
                llm=ChatBot(llm_provider="gemini", model_name=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-001")),
                available_tools=ToolManager([SearchStocksTool()])
            )
            logger.info("✅ SpoonScoutAgent initialized with Gemini LLM")
        
        def _interpret_query_to_criteria(self, query: str, default_limit: int = 30) -> Dict[str, Any]:
            """Interpret a natural language query into search criteria.
            Uses simple heuristics as fallback when LLM tooling/config is unavailable."""
            q = (query or "").lower()
            criteria: Dict[str, Any] = {"limit": default_limit}
            if any(k in q for k in ["tech", "technology"]):
                criteria["sector"] = "Technology"
            import re
            m_growth = re.search(r"(growth|revenue\s*growth)[^0-9]*([0-9]{1,2})%", q)
            if m_growth:
                criteria["min_revenue_growth"] = float(m_growth.group(2))
            elif "growth" in q:
                criteria["min_revenue_growth"] = 10.0
            m_rsi = re.search(r"rsi[^0-9]*([0-9]{1,2})", q)
            if m_rsi:
                criteria["max_rsi"] = float(m_rsi.group(1))
            elif any(k in q for k in ["beaten down", "undervalued", "oversold"]):
                criteria["max_rsi"] = 50.0
            # Market cap parsing (e.g., 50B, 0.5T, billion/trillion words)
            m_cap_b = re.search(r"([0-9]+)\s*[bB](?:illion)?", q)
            m_cap_t = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[tT](?:rillion)?", q)
            if m_cap_t:
                criteria["min_market_cap"] = float(m_cap_t.group(1)) * 1_000_000_000_000
            elif m_cap_b:
                criteria["min_market_cap"] = float(m_cap_b.group(1)) * 1_000_000_000
            elif "large cap" in q or ("large" in q and ("tech" in q or "technology" in q)):
                criteria["min_market_cap"] = 10_000_000_000
            return criteria

        async def _llm_generate_criteria(self, query: str, default_limit: int = 30) -> Dict[str, Any]:
            if not query:
                return {}
            prompt = (
                "Based on the description of the Query, please generate a stock screener criteria and return ONLY JSON with keys: "
                "min_market_cap (USD number), max_rsi (number), min_revenue_growth (percent number), sector (only populate the value "
                " if the query clearly indicates a sector, for example: Find large cap tech stocks should return Technology as the sector value. string from this list only: "
                "Technology, Consumer Discretionary, Financials, Healthcare, Energy, Consumer Staples, Communication Services, Utilities, Materials, Real Estate, Information Technology, Industrials). "
                "Query: " + query
            )
            try:
                result_text = None
                logger.debug(f"🔍 Prompt: {prompt}")
                if hasattr(self.llm, "ask"):
                    try:
                        msgs = [Message(role="user", content=prompt)]  # type: ignore
                    except Exception:
                        msgs = [{"role": "user", "content": prompt}]
                    rt = self.llm.ask(msgs)
                    if asyncio.iscoroutine(rt):
                        result_text = await rt
                    else:
                        result_text = rt
                elif hasattr(self.llm, "chat"):
                    try:
                        sys = Message(role="system", content="Return JSON only.")  # type: ignore
                        usr = Message(role="user", content=prompt)  # type: ignore
                        payload = [sys, usr]
                    except Exception:
                        payload = [
                            {"role": "system", "content": "Return JSON only."},
                            {"role": "user", "content": prompt}
                        ]
                    result = self.llm.chat(payload)
                    if asyncio.iscoroutine(result):
                        result = await result
                    if isinstance(result, dict) and "content" in result:
                        result_text = result["content"]
                    else:
                        result_text = str(result)
                else:
                    logger.warning(f"❌ No result_text generated, self.llm: {self.llm}")
                    return {}
                if not result_text:
                    logger.warning(f"❌ No result_text generated: {result_text}")
                    return {}
                
                import re
                m = re.search(r"\{[\s\S]*\}", result_text)
                text = m.group(0) if m else result_text
                data = json.loads(text)
                out: Dict[str, Any] = {"limit": default_limit}
                allowed_sectors = {
                    "Technology",
                    "Consumer Discretionary",
                    "Financials",
                    "Healthcare",
                    "Energy",
                    "Consumer Staples",
                    "Communication Services",
                    "Utilities",
                    "Materials",
                    "Real Estate",
                    "Information Technology",
                    "Industrials",
                }
                if isinstance(data, dict):
                    if "min_market_cap" in data and data["min_market_cap"] is not None:
                        out["min_market_cap"] = float(data["min_market_cap"])
                    if "max_rsi" in data and data["max_rsi"] is not None:
                        out["max_rsi"] = float(data["max_rsi"])
                    if "min_revenue_growth" in data and data["min_revenue_growth"] is not None:
                        out["min_revenue_growth"] = float(data["min_revenue_growth"])
                    if "sector" in data and data["sector"] is not None:
                        sec = str(data["sector"]).strip()
                        if sec in allowed_sectors:
                            out["sector"] = sec
                return out
            except Exception as e:
                logger.error(f"❌ Error in _llm_generate_criteria: {e}")
                import traceback
                traceback.print_exc()
                return {}

        async def run(self, request: str) -> str:  # type: ignore
            state = json.loads(request) if isinstance(request, str) else request
            natural_query = state.get("natural_query", "")
            criteria = state.get("search_criteria", {})
            limit = state.get("stock_count", criteria.get("limit", 20))
            logger.debug(f"natural_query: {natural_query}")
            logger.debug(f"criteria: {criteria}")
            criteria = await self._llm_generate_criteria(natural_query, default_limit=limit)
            logger.debug(f"generated: {criteria}")
            if natural_query and not criteria:
                criteria = self._interpret_query_to_criteria(natural_query, default_limit=limit)
                logger.debug(f"criteria created: {criteria}")
            
            params = {
                "limit": limit,
                "min_market_cap": criteria.get("min_market_cap", 0),
                "max_rsi": criteria.get("max_rsi", 100),
                "min_revenue_growth": criteria.get("min_revenue_growth", 0),
                "sector": criteria.get("sector", "")
            }
            data = await self.available_tools.tools[0].execute(**params)
            return json.dumps({"candidates": data, "search_criteria": params})

    class SpoonEvaluationAgent(ToolCallAgent):
        def __init__(self):
            super().__init__(
                llm=ChatBot(llm_provider="gemini", model_name=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-001")),
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
                llm=ChatBot(llm_provider="gemini", model_name=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-001")),
                available_tools=ToolManager([SentimentTool()])
            )
            logger.info("✅ SpoonSentimentAgent initialized with Gemini LLM")
        async def run(self, request: str) -> str:  # type: ignore
            state = json.loads(request) if isinstance(request, str) else request
            tickers = state.get("stocks_to_analyze", [])
            scores = await self.available_tools.tools[0].execute(tickers=tickers)
            return json.dumps({"sentiment_scores": scores})

    class SpoonRankingAgent(ToolCallAgent):
        def __init__(self):
            super().__init__(
                llm=ChatBot(llm_provider="gemini", model_name=os.getenv("GEMINI_MODEL", "models/gemini-1.5-pro-001")),
                available_tools=ToolManager([RankTool()])
            )
            logger.info("✅ SpoonRankingAgent initialized with Gemini LLM")
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
    natural_query: str


class ScoutState(TypedDict):
    """State for ScoutAgent"""
    candidates: List[Dict[str, Any]]
    search_criteria: Dict[str, Any]
    stock_count: int
    natural_query: str


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
        logger.error(f"Error fetching data for {ticker}: {e}")
    
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
    logger.info(f"Searching stocks with criteria: {criteria}")
    
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "limit": criteria.get("limit", 30),
                "min_market_cap": criteria.get("min_market_cap", 0),
                "max_rsi": criteria.get("max_rsi", 100),
                "min_revenue_growth": criteria.get("min_revenue_growth", 0),
                "sector": criteria.get("sector", "")
            }
            
            response = await client.get("http://localhost:8001/api/stocks/search", params=params)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API returned {len(data)} stocks")
                return data
    except Exception as e:
        logger.error(f"Error searching stocks via API: {e}")
    
    # Fallback: return mock S&P 500 stocks
    logger.warning("Using fallback mock data")
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
    
    logger.info(f"Generated {len(candidates)} mock candidates")
    return candidates


# Node implementations
async def scout_node(state: ScoutState) -> Dict[str, Any]:
    """Scout node to find stock candidates"""
    logger.info(f"🔍 ScoutAgent: Searching for {state['stock_count']} stock candidates...")
    
    criteria = state.get("search_criteria", {})
    criteria["limit"] = state["stock_count"]
    
    candidates = await search_stocks_from_api(criteria)
    
    logger.info(f"✅ ScoutAgent: Found {len(candidates)} candidates")
    
    return {
        "candidates": candidates,
        "search_criteria": criteria
    }


async def evaluation_node(state: EvaluationState) -> Dict[str, Any]:
    """Evaluation node to analyze fundamentals and technical data"""
    logger.info(f"📊 EvaluationAgent: Evaluating {len(state['stocks_to_evaluate'])} stocks...")
    
    evaluation_results = []
    
    for stock in state["stocks_to_evaluate"]:
        ticker = stock["ticker"]
        logger.debug(f"  Evaluating {ticker}...")
        
        # Fetch detailed data
        detailed_data = await fetch_stock_data_from_api(ticker)
        logger.debug(f"    Data: RSI={detailed_data['rsi']}, Price=${detailed_data['current_price']}, MA20=${detailed_data['ma_20']}, MA50=${detailed_data['ma_50']}")
        
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
        logger.debug(f"    DEBUG: Checking criteria - Fundamental={fundamental_score:.2f} >= 6.0? {fundamental_score >= 6.0}")
        logger.debug(f"    DEBUG: Checking criteria - Technical={technical_score:.2f} >= 5.0? {technical_score >= 5.0}")
        logger.debug(f"    DEBUG: Checking criteria - Market Cap={market_cap:,.0f} > 10B? {market_cap > 10000000000}")
        
        evaluation_passed = (
            fundamental_score >= 6.0 and  # Good fundamental score
            technical_score >= 5.0 and   # Decent technical score
            market_cap > 10000000000  # Large cap (>10B)
        )
        logger.info(f"    Scores: Fundamental={fundamental_score:.2f}, Technical={technical_score:.2f}, Market Cap={market_cap:,.0f}, Passed={evaluation_passed}")
        
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
    logger.info(f"✅ EvaluationAgent: {len(passed_stocks)} stocks passed evaluation out of {len(evaluation_results)}")
    logger.debug(f"📊 Sample evaluation results: {evaluation_results[:3] if evaluation_results else 'None'}")
    
    return {
        "evaluation_results": evaluation_results,
        "passed_stocks": passed_stocks
    }


async def sentiment_node(state: SentimentState) -> Dict[str, Any]:
    """Sentiment node to analyze news and social media sentiment"""
    logger.info(f"💭 SentimentAgent: Analyzing sentiment for {len(state['stocks_to_analyze'])} stocks...")
    
    sentiment_scores = {}
    
    for ticker in state["stocks_to_analyze"]:
        try:
            logger.debug(f"🔍 Analyzing sentiment for {ticker}...")
            
            # For now, use random sentiment scores to test the pipeline
            # In a real implementation, this would fetch news and analyze sentiment
            sentiment_score = random.uniform(4, 9)  # Random sentiment between 4-9
            sentiment_scores[ticker] = round(sentiment_score, 2)
            logger.info(f"💯 Sentiment score for {ticker}: {sentiment_scores[ticker]:.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment for {ticker}: {e}")
            sentiment_scores[ticker] = random.uniform(4, 7)
            logger.warning(f"⚠️ Using fallback sentiment for {ticker}: {sentiment_scores[ticker]:.2f}")
    
    logger.info(f"✅ SentimentAgent: Completed sentiment analysis for {len(sentiment_scores)} stocks")
    logger.debug(f"📊 Sentiment scores: {sentiment_scores}")
    
    return {
        "sentiment_scores": sentiment_scores
    }


async def ranking_node(state: RankingState) -> Dict[str, Any]:
    """Ranking node to score stocks with weighted algorithm"""
    logger.info(f"🏆 RankingAgent: Ranking {len(state['evaluated_stocks'])} stocks...")
    logger.info(f"📊 Available sentiment scores: {len(state.get('sentiment_scores', {}))} stocks")
    
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
        
        logger.info(f"📈 {ticker}: Fundamental={fundamental_score}, Technical={technical_score}, Sentiment={sentiment_score:.2f}, Weighted={weighted_score:.2f}")
    
    # Sort by weighted score (descending)
    final_scores.sort(key=lambda x: x["weighted_score"], reverse=True)
    
    logger.info(f"✅ RankingAgent: Completed ranking of {len(final_scores)} stocks")
    logger.info(f"🏆 Top 5 ranked stocks: {[stock['ticker'] for stock in final_scores[:5]]}")
    
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
    scout_agent = create_scout_agent()
    eval_agent = GraphAgent("evaluation", create_evaluation_graph())
    sentiment_agent = GraphAgent("sentiment", create_sentiment_graph())
    ranking_agent = GraphAgent("ranking", create_ranking_graph())
    
    # Define orchestrator nodes
    async def initialize_portfolio(state: TradingState) -> Dict[str, Any]:
        """Initialize portfolio building process"""
        logger.info("🚀 Initializing portfolio building process...")
        return {
            "current_step": "initialized",
            "iteration_count": 0,
            "target_portfolio_size": 20
        }
    
    async def scout_stocks(state: TradingState) -> Dict[str, Any]:
        """Scout for stock candidates"""
        logger.info(f"🔍 Scouting iteration {state['iteration_count'] + 1}...")
        
        # Determine how many stocks to scout
        remaining_needed = state["target_portfolio_size"] - len(state["evaluated_stocks"])
        scout_count = max(20, remaining_needed * 2)  # Scout more than needed
        
        scout_state = ScoutState(
            candidates=[],
            search_criteria={},
            stock_count=scout_count
        )
        state_payload = {**scout_state, "natural_query": state.get("natural_query", "")}  # type: ignore
        
        result = await scout_agent.run(json.dumps(state_payload))
        scout_result = json.loads(result) if isinstance(result, str) else result
        
        return {
            "scout_candidates": scout_result.get("candidates", []),
            "current_step": "scouting_completed"
        }
    
    async def evaluate_stocks(state: TradingState) -> Dict[str, Any]:
        """Evaluate scouted stocks"""
        logger.info(f"📊 Evaluating {len(state['scout_candidates'])} candidates...")
        
        eval_state = EvaluationState(
            stocks_to_evaluate=state["scout_candidates"],
            evaluation_results=[],
            fundamental_metrics={},
            technical_metrics={}
        )
        
        logger.info(f"🔄 Running evaluation agent...")
        result = await eval_agent.run(json.dumps(eval_state))
        logger.debug(f"📋 Evaluation result: {result}")
        
        eval_result = json.loads(result) if isinstance(result, str) else result
        
        passed_stocks = eval_result.get("passed_stocks", [])
        logger.info(f"✅ {len(passed_stocks)} stocks passed evaluation")
        logger.debug(f"📋 First few passed stocks: {passed_stocks[:2] if passed_stocks else 'None'}")
        
        # Combine with existing evaluated stocks
        all_evaluated = state["evaluated_stocks"] + passed_stocks
        
        return {
            "evaluated_stocks": all_evaluated,
            "current_step": "evaluation_completed"
        }
    
    async def analyze_sentiment(state: TradingState) -> Dict[str, Any]:
        """Analyze sentiment for evaluated stocks"""
        tickers = [stock["ticker"] for stock in state["evaluated_stocks"]]
        logger.info(f"💭 Analyzing sentiment for {len(tickers)} stocks...")
        
        sentiment_state = SentimentState(
            stocks_to_analyze=tickers,
            news_data={},
            social_data={},
            sentiment_scores={}
        )
        
        logger.debug(f"🎯 Calling sentiment agent with state: {sentiment_state}")
        result = await sentiment_agent.run(json.dumps(sentiment_state))
        sentiment_result = json.loads(result) if isinstance(result, str) else result
        
        logger.debug(f"📊 Sentiment analysis result: {sentiment_result}")
        sentiment_scores = sentiment_result.get("sentiment_scores", {})
        logger.info(f"💯 Sentiment scores: {sentiment_scores}")
        
        return {
            "sentiment_scores": sentiment_scores,
            "current_step": "sentiment_completed"
        }
    
    async def rank_stocks(state: TradingState) -> Dict[str, Any]:
        """Rank evaluated stocks"""
        logger.info(f"🏆 Ranking {len(state['evaluated_stocks'])} stocks...")
        logger.info(f"📊 Available sentiment scores: {state.get('sentiment_scores', {})}")
        
        ranking_state = RankingState(
            evaluated_stocks=state["evaluated_stocks"],
            sentiment_scores=state["sentiment_scores"],
            final_scores=[],
            weights={"fundamental": 5, "technical": 3, "sentiment": 2}
        )
        
        logger.debug(f"🎯 Calling ranking agent with state: {len(ranking_state['evaluated_stocks'])} stocks")
        result = await ranking_agent.run(json.dumps(ranking_state))
        ranking_result = json.loads(result) if isinstance(result, str) else result
        
        logger.debug(f"📊 Ranking result: {ranking_result}")
        final_rankings = ranking_result.get("final_scores", [])
        logger.info(f"💯 Final rankings count: {len(final_rankings)}")
        
        return {
            "final_rankings": final_rankings,
            "current_step": "ranking_completed"
        }
    
    async def build_portfolio(state: TradingState) -> Dict[str, Any]:
        """Build final portfolio"""
        logger.info("📈 Building final portfolio...")
        
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
            logger.info(f"✅ Portfolio complete with {current_size} stocks")
            return {"current_step": "completed"}
        else:
            logger.info(f"📈 Need {target_size - current_size} more stocks")
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