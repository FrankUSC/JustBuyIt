import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GlassCard } from './GlassCard';
import { GradientText } from './GradientText';
import { ProgressBar } from './ProgressBar';
import { useAgentStream } from '../stores/agentStream';
import { Search, TrendingUp, Activity, Brain, BarChart3, MessageSquare, Trophy } from 'lucide-react';

interface AgentPhase {
  name: string;
  description: string;
  icon: React.ReactNode;
  status: 'pending' | 'running' | 'completed' | 'error';
}

export const SpoonAIAgentOrchestrator: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentPhase, setCurrentPhase] = useState(0);
  const [portfolio, setPortfolio] = useState<any[]>([]);
  const [iterations, setIterations] = useState(0);
  const [agentsStatus, setAgentsStatus] = useState<any>({});
  const [scoutOutput, setScoutOutput] = useState<any>(null);
  const [evalOutput, setEvalOutput] = useState<any>(null);
  const [sentimentOutput, setSentimentOutput] = useState<any>(null);
  const [rankingOutput, setRankingOutput] = useState<any>(null);
  const [activeTest, setActiveTest] = useState<null | 'scout' | 'evaluation' | 'sentiment' | 'ranking'>(null);
  
  const { addLog, setScanning, logs } = useAgentStream();

  const initialPhases: AgentPhase[] = [
    { 
      name: 'ScoutAgent', 
      description: 'Finding 20+ stock candidates from S&P 500...',
      icon: <Search className="w-5 h-5" />,
      status: 'pending'
    },
    { 
      name: 'EvaluationAgent', 
      description: 'Analyzing fundamentals and technical indicators...',
      icon: <BarChart3 className="w-5 h-5" />,
      status: 'pending'
    },
    { 
      name: 'SentimentAgent', 
      description: 'Pulling latest news and X tweets for sentiment scoring...',
      icon: <MessageSquare className="w-5 h-5" />,
      status: 'pending'
    },
    { 
      name: 'RankingAgent', 
      description: 'Scoring stocks with weighted algorithm (Fundamentals:5, Technical:3, Sentiment:2)...',
      icon: <Trophy className="w-5 h-5" />,
      status: 'pending'
    },
    { 
      name: 'Portfolio Builder', 
      description: 'Constructing final portfolio with top 20 stocks...',
      icon: <Brain className="w-5 h-5" />,
      status: 'pending'
    }
  ];
  const [phases, setPhases] = useState<AgentPhase[]>(initialPhases);

  // Check agents status on mount
  useEffect(() => {
    checkAgentsStatus();
  }, []);

  const checkAgentsStatus = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/agents/status');
      const status = await response.json();
      setAgentsStatus(status);
    } catch (error) {
      console.error('Failed to check agents status:', error);
    }
  };

  const runSpoonAIPipeline = async () => {
    if (isRunning) return;
    
    setIsRunning(true);
    setCurrentPhase(0);
    setScanning(true);
    setPortfolio([]);
    setIterations(0);
    setPhases(initialPhases);
    
    try {
      addLog('SPOONAI', '🚀 Starting multi-agent portfolio building process...', 'info');
      setPhases((prev) => prev.map((p, i) => ({ ...p, status: i === 0 ? 'running' : p.status })));
      
      // Run the main orchestrator
      const response = await fetch('http://localhost:8001/api/agents/build-portfolio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setPhases((prev) => prev.map((p) => ({ ...p, status: 'completed' })));
      
      setPortfolio(result.portfolio);
      setIterations(result.iterations);
      
      addLog('SPOONAI', `✅ Portfolio building completed with ${result.total_stocks} stocks`, 'success');
      addLog('SPOONAI', `📊 Final portfolio: ${result.portfolio.map((stock: any) => stock.ticker).join(', ')}`, 'info');
      
    } catch (error) {
      console.error('SpoonAI Pipeline failed:', error);
      addLog('SPOONAI', '❌ Multi-agent pipeline execution failed', 'error');
      setPhases((prev) => prev.map((p, i) => ({ ...p, status: i === Math.max(0, currentPhase - 1) ? 'error' : p.status })));
    } finally {
      setIsRunning(false);
      setScanning(false);
      setCurrentPhase(0);
    }
  };

  

  const testIndividualAgent = async (agentType: string) => {
    try {
      addLog('SPOONAI', `🧪 Testing ${agentType}...`, 'info');
      setActiveTest(agentType as 'scout' | 'evaluation' | 'sentiment' | 'ranking');
      
      let response;
      let result;
      
      switch (agentType) {
        case 'scout':
          response = await fetch('http://localhost:8001/api/agents/scout?stock_count=10', {
            method: 'POST'
          });
          result = await response.json();
          addLog('SCOUT', `Found ${result.total_found} candidates`, 'success');
          setScoutOutput({
            total_found: result.total_found,
            sample: (result.candidates || []).slice(0, 5).map((c: any) => c.ticker)
          });
          await fetch('http://localhost:8001/api/memory/append', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: 'phf', role: 'assistant', content: `Scout found ${result.total_found}`, meta: { type: 'scout', sample: (result.candidates || []).slice(0,5).map((c:any)=>c.ticker) } })
          });
          break;
        case 'evaluation': {
          response = await fetch('http://localhost:8001/api/agents/scout?stock_count=8', { method: 'POST' });
          const scoutForEval = await response.json();
          result = await fetch('http://localhost:8001/api/agents/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(scoutForEval.candidates || [])
          });
          const evalRes = await result.json();
          addLog('EVAL', `Evaluated ${(evalRes.evaluation_results || []).length} and ${evalRes.passed_stocks?.length || 0} passed`, 'success');
          setEvalOutput({
            total_evaluated: (evalRes.evaluation_results || []).length,
            total_passed: (evalRes.passed_stocks || []).length,
            sample: (evalRes.passed_stocks || []).slice(0, 5).map((p: { ticker: string; fundamental_score: number; technical_score: number }) => ({
              ticker: p.ticker,
              f: p.fundamental_score,
              t: p.technical_score
            }))
          });
          await fetch('http://localhost:8001/api/memory/append', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: 'phf', role: 'assistant', content: `Evaluation passed ${(evalRes.passed_stocks||[]).length}`, meta: { type: 'evaluation', sample: (evalRes.passed_stocks||[]).slice(0,5).map((p:any)=>p.ticker) } })
          });
          break;
        }
        case 'sentiment': {
          response = await fetch('http://localhost:8001/api/agents/sentiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(['AAPL', 'MSFT', 'GOOGL'])
          });
          result = await response.json();
          addLog('SENTIMENT', `Analyzed sentiment for ${result.total_analyzed} stocks`, 'success');
          const scores = result.sentiment_scores as Record<string, number>;
          setSentimentOutput({
            total_analyzed: result.total_analyzed,
            sample: Object.entries(scores || {}).slice(0, 5).map(([k, v]) => ({ ticker: k, score: v }))
          });
          await fetch('http://localhost:8001/api/memory/append', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: 'phf', role: 'assistant', content: `Sentiment analyzed ${result.total_analyzed}`, meta: { type: 'sentiment', sample: Object.entries(scores||{}).slice(0,5) } })
          });
          break;
        }
        case 'ranking': {
          response = await fetch('http://localhost:8001/api/agents/scout?stock_count=10', { method: 'POST' });
          const scoutForRank = await response.json();
          const evalResp = await fetch('http://localhost:8001/api/agents/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(scoutForRank.candidates || [])
          });
          const evalData = await evalResp.json();
          const tickers = (evalData.passed_stocks || []).map((s: { ticker: string }) => s.ticker);
          const sentResp = await fetch('http://localhost:8001/api/agents/sentiment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tickers)
          });
          const sentData = await sentResp.json();
          const rankPayload = {
            evaluated_stocks: evalData.passed_stocks || [],
            sentiment_scores: (sentData.sentiment_scores || {}) as Record<string, number>,
            weights: { fundamental: 5, technical: 3, sentiment: 2 }
          };
          const rankResp = await fetch('http://localhost:8001/api/agents/rank', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rankPayload)
          });
          const rankData = await rankResp.json();
          addLog('RANK', `Ranked ${(rankData.final_rankings || []).length} stocks`, 'success');
          setRankingOutput({
            total_ranked: (rankData.final_rankings || []).length,
            sample: (rankData.final_rankings || []).slice(0, 5).map((r: { ticker: string; weighted_score: number }) => ({ ticker: r.ticker, score: r.weighted_score }))
          });
          await fetch('http://localhost:8001/api/memory/append', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_id: 'phf', role: 'assistant', content: `Ranking completed ${(rankData.final_rankings||[]).length}`, meta: { type: 'ranking', sample: (rankData.final_rankings||[]).slice(0,5).map((r:any)=>r.ticker) } })
          });
          break;
        }
        default:
          addLog('SPOONAI', `Testing ${agentType} not implemented yet`, 'warning');
      }
      
    } catch (error) {
      console.error(`Testing ${agentType} failed:`, error);
      addLog('SPOONAI', `❌ Testing ${agentType} failed`, 'error');
    }
  };

  return (
    <div className="space-y-6">
      {/* Agent Status */}
      <GlassCard className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-white">SpoonAI Agent Status</h3>
          <Brain className="w-6 h-6 text-blue-400" />
        </div>
        <div className="grid grid-cols-5 gap-4">
          {Object.entries(agentsStatus).map(([agent, status]) => (
            <div
              key={agent}
              className={`p-3 rounded-lg text-center transition-all duration-300 ${
                status 
                  ? 'bg-emerald-600/20 border border-emerald-600/30' 
                  : 'bg-red-600/20 border border-red-600/30'
              }`}
            >
              <div className="text-xs font-medium text-white mb-1">
                {agent.replace('_', ' ').toUpperCase()}
              </div>
              <div className={`text-xs ${
                status ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {status ? 'Ready' : 'Offline'}
              </div>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Main Control */}
      <GlassCard className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-xl font-bold text-white mb-2">Multi-Agent Portfolio Builder</h3>
            <p className="text-slate-400 text-sm">
              AI-powered stock selection using Scout → Evaluation → Sentiment → Ranking pipeline
            </p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={() => testIndividualAgent('scout')}
              disabled={isRunning || !agentsStatus.scout_agent}
              className="px-4 py-2 bg-slate-700/50 text-white text-sm rounded-lg hover:bg-slate-600/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
            >Test Scout</button>
            <button
              onClick={() => testIndividualAgent('evaluation')}
              disabled={isRunning || !agentsStatus.evaluation_agent}
              className="px-4 py-2 bg-slate-700/50 text-white text-sm rounded-lg hover:bg-slate-600/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
            >Test Evaluation</button>
            <button
              onClick={() => testIndividualAgent('sentiment')}
              disabled={isRunning || !agentsStatus.sentiment_agent}
              className="px-4 py-2 bg-slate-700/50 text-white text-sm rounded-lg hover:bg-slate-600/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
            >
              Test Sentiment
            </button>
            <button
              onClick={() => testIndividualAgent('ranking')}
              disabled={isRunning || !agentsStatus.ranking_agent}
              className="px-4 py-2 bg-slate-700/50 text-white text-sm rounded-lg hover:bg-slate-600/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
            >Test Ranking</button>
            <button
              onClick={runSpoonAIPipeline}
              disabled={isRunning || !agentsStatus.trading_orchestrator}
              className="px-6 py-3 bg-gradient-to-r from-blue-600 to-violet-600 text-white font-medium rounded-xl hover:from-blue-700 hover:to-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center space-x-2"
            >
              {isRunning ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  <span>Building...</span>
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4" />
                  <span>Run Multi-Agent Pipeline</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Progress Indicator */}
        {isRunning && (
          <div className="mb-6">
            <ProgressBar 
              current={currentPhase} 
              total={phases.length} 
              label="Agent Pipeline Progress"
            />
          </div>
        )}

        {/* Agent Phases */}
        <div className="space-y-3">
          {phases.filter((_, i) => activeTest === null ? true : (
            (activeTest === 'scout' && i === 0) ||
            (activeTest === 'evaluation' && i === 1) ||
            (activeTest === 'sentiment' && i === 2) ||
            (activeTest === 'ranking' && i === 3)
          )).map((phase, index) => (
            <motion.div
              key={phase.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-4 rounded-xl border transition-all duration-300 ${
                phase.status === 'completed'
                  ? 'bg-emerald-600/10 border-emerald-600/30'
                  : phase.status === 'error'
                  ? 'bg-red-600/10 border-red-600/30'
                  : phase.status === 'running'
                  ? 'bg-blue-600/10 border-blue-600/30 animate-pulse'
                  : 'bg-slate-800/30 border-white/10'
              }`}
            >
              <div className="flex items-center space-x-4">
                <div className={`p-2 rounded-lg ${
                  phase.status === 'completed'
                    ? 'bg-emerald-600/20 text-emerald-400'
                    : phase.status === 'error'
                    ? 'bg-red-600/20 text-red-400'
                    : phase.status === 'running'
                    ? 'bg-blue-600/20 text-blue-400'
                    : 'bg-slate-700/50 text-slate-400'
                }`}>
                  {phase.icon}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h4 className="text-white font-medium">{phase.name}</h4>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      phase.status === 'completed'
                        ? 'bg-emerald-600/20 text-emerald-400'
                        : phase.status === 'error'
                        ? 'bg-red-600/20 text-red-400'
                        : phase.status === 'running'
                        ? 'bg-blue-600/20 text-blue-400'
                        : 'bg-slate-600/20 text-slate-400'
                    }`}>
                      {phase.status === 'completed' ? '✅ Completed' :
                       phase.status === 'error' ? '❌ Error' :
                       phase.status === 'running' ? '🔄 Running' : '⏳ Pending'}
                    </span>
                  </div>
                  <p className="text-slate-400 text-sm mt-1">{phase.description}</p>
                  {index === 0 && logs && (
                    <div className="mt-2 text-xs text-slate-300 bg-slate-900/30 rounded-lg p-2 max-h-64 overflow-y-auto">
                      {logs.filter(l=>l.agent==='SCOUT').map((l)=> (
                        <div key={l.id}>[{l.type}] {l.message}</div>
                      ))}
                    </div>
                  )}
                  {index === 1 && logs && (
                    <div className="mt-2 text-xs text-slate-300 bg-slate-900/30 rounded-lg p-2 max-h-64 overflow-y-auto">
                      {logs.filter(l=>l.agent==='EVAL').map((l)=> (
                        <div key={l.id}>[{l.type}] {l.message}</div>
                      ))}
                    </div>
                  )}
                  {index === 2 && logs && (
                    <div className="mt-2 text-xs text-slate-300 bg-slate-900/30 rounded-lg p-2 max-h-64 overflow-y-auto">
                      {logs.filter(l=>l.agent==='SENTIMENT').map((l)=> (
                        <div key={l.id}>[{l.type}] {l.message}</div>
                      ))}
                    </div>
                  )}
                  {index === 3 && logs && (
                    <div className="mt-2 text-xs text-slate-300 bg-slate-900/30 rounded-lg p-2 max-h-64 overflow-y-auto">
                      {logs.filter(l=>l.agent==='RANK').map((l)=> (
                        <div key={l.id}>[{l.type}] {l.message}</div>
                      ))}
                    </div>
                  )}
                  {index === 4 && logs && (
                    <div className="mt-2 text-xs text-slate-300 bg-slate-900/30 rounded-lg p-2 max-h-64 overflow-y-auto">
                      {logs.filter(l=>l.agent==='PORTFOLIO').map((l)=> (
                        <div key={l.id}>[{l.type}] {l.message}</div>
                      ))}
                    </div>
                  )}
                  {index === 0 && scoutOutput && (
                    <div className="mt-2 text-xs text-slate-300">
                      <div>Found: {scoutOutput.total_found}</div>
                      <div>Sample: {scoutOutput.sample.join(', ')}</div>
                    </div>
                  )}
                  {index === 1 && evalOutput && (
                    <div className="mt-2 text-xs text-slate-300">
                      <div>Evaluated: {evalOutput.total_evaluated}</div>
                      <div>Passed: {evalOutput.total_passed}</div>
                      <div>Sample: {evalOutput.sample.map((s: any) => `${s.ticker} (F:${s.f}, T:${s.t})`).join(', ')}</div>
                    </div>
                  )}
                  {index === 2 && sentimentOutput && (
                    <div className="mt-2 text-xs text-slate-300">
                      <div>Analyzed: {sentimentOutput.total_analyzed}</div>
                      <div>Sample: {sentimentOutput.sample.map((s: any) => `${s.ticker}:${s.score}`).join(', ')}</div>
                    </div>
                  )}
                  {index === 3 && rankingOutput && (
                    <div className="mt-2 text-xs text-slate-300">
                      <div>Ranked: {rankingOutput.total_ranked}</div>
                      <div>Top: {rankingOutput.sample.map((s: any) => `${s.ticker}:${s.score}`).join(', ')}</div>
                    </div>
                  )}
                  {index === 4 && portfolio.length > 0 && (
                    <div className="mt-2 text-xs text-slate-300">
                      <div>Total: {portfolio.length}</div>
                      <div>Sample: {portfolio.slice(0,5).map((s:any)=>`${s.ticker}:${(s.weight*100).toFixed(1)}%`).join(', ')}</div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Iteration Counter */}
        {iterations > 0 && (
          <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
            <div className="text-center">
              <p className="text-slate-400 text-sm">Iterations Required</p>
              <GradientText text={`${iterations} iterations`} className="text-lg font-bold" />
              <p className="text-slate-500 text-xs mt-1">
                System scouted additional stocks to reach target portfolio size
              </p>
            </div>
          </div>
        )}
      </GlassCard>

      {/* Results */}
      <AnimatePresence>
        {portfolio.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            <GlassCard className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-white">Generated Portfolio</h3>
                <TrendingUp className="w-6 h-6 text-emerald-400" />
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                {portfolio.map((stock, index) => (
                  <div key={stock.ticker} className="p-4 bg-slate-800/50 rounded-lg border border-white/10">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-bold">{stock.ticker}</span>
                      <span className="text-slate-400 text-sm">#{index + 1}</span>
                    </div>
                    <div className="text-slate-400 text-sm mb-1">
                      Sector: {stock.sector}
                    </div>
                    <div className="text-slate-400 text-sm mb-1">
                      Price: ${stock.current_price?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-slate-400 text-sm">
                      Score: {stock.score?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="text-slate-400 text-sm">
                      Weight: {(stock.weight * 100)?.toFixed(1) || 'N/A'}%
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="grid grid-cols-3 gap-6 text-center">
                <div>
                  <p className="text-slate-400 text-sm mb-2">Total Stocks</p>
                  <GradientText text={portfolio.length.toString()} className="text-2xl font-bold" />
                </div>
                <div>
                  <p className="text-slate-400 text-sm mb-2">Equal Weight</p>
                  <GradientText text={`${(100 / portfolio.length).toFixed(1)}%`} className="text-2xl font-bold" />
                </div>
                <div>
                  <p className="text-slate-400 text-sm mb-2">Avg Score</p>
                  <GradientText text={`${(portfolio.reduce((sum, stock) => sum + (stock.score || 0), 0) / portfolio.length).toFixed(2)}`} className="text-2xl font-bold" />
                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};