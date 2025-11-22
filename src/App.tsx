import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { HeroMetrics } from './components/HeroMetrics';
import { AgentStream } from './components/AgentStream';
import { GlassCard } from './components/GlassCard';
import { GradientText } from './components/GradientText';
import { AgentOrchestrator } from './components/AgentOrchestrator';
import { SpoonAIAgentOrchestrator } from './components/SpoonAIAgentOrchestrator';
import { PortfolioChart } from './components/PortfolioChart';
import { motion } from 'framer-motion';
import { usePortfolioManager } from './stores/portfolioManager';
import { useBacktestAgent } from './stores/backtestAgent';

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const { portfolio: selectedPortfolio, setTargetPositions, target_positions } = usePortfolioManager();
  const { results: backtestResults, runBacktest, isLoading: backtestLoading, currentStep, totalSteps, setStepMonths, stepMonths } = useBacktestAgent();
  const [backtestStart, setBacktestStart] = useState('2023-01-01');
  const [backtestEnd, setBacktestEnd] = useState('2025-01-01');
  const [summary, setSummary] = useState<string>('');

  

  

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="flex">
        <Sidebar activeView={activeView} onViewChange={setActiveView} />
        
        <div className="flex-1 flex">
          {/* Main Content */}
          <div className="flex-1 p-8">
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
            >
              <h1 className="text-4xl font-bold text-white mb-2">
                <GradientText text="Pocket Hedge Fund" />
              </h1>
              <p className="text-slate-400 text-lg">Autonomous AI Quantitative Trading Platform</p>
            </motion.div>

            {/* Agent Orchestrator */}
            {activeView === 'dashboard' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="mb-8"
              >
                <AgentOrchestrator />
              </motion.div>
            )}

            {/* SpoonAI Agent Orchestrator */}
            {activeView === 'spoonai' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="mb-8"
              >
                <SpoonAIAgentOrchestrator />
              </motion.div>
            )}

            {/* Backtest */}
            {activeView === 'backtest' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="mb-8 space-y-6"
              >
                <GlassCard className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-white">Selected Stocks</h3>
                  </div>
                  {selectedPortfolio.length === 0 ? (
                    <div className="text-slate-400 text-sm">No stock candidates. Start with SpoonAI Agent page first.</div>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {selectedPortfolio.map((s) => (
                        <span key={s.ticker} className="px-3 py-1 bg-slate-800/50 border border-white/10 rounded-lg text-white text-xs">{s.ticker}</span>
                      ))}
                    </div>
                  )}
                </GlassCard>

                <GlassCard className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                    <div>
                      <label className="text-slate-400 text-xs">Start Date</label>
                      <input type="date" value={backtestStart} onChange={(e)=>setBacktestStart(e.target.value)} className="w-full px-3 py-2 bg-slate-800/50 border border-white/10 rounded-lg text-white" />
                    </div>
                    <div>
                      <label className="text-slate-400 text-xs">End Date</label>
                      <input type="date" value={backtestEnd} onChange={(e)=>setBacktestEnd(e.target.value)} className="w-full px-3 py-2 bg-slate-800/50 border border-white/10 rounded-lg text-white" />
                    </div>
                    <div>
                      <label className="text-slate-400 text-xs">Rebalance Interval (months)</label>
                      <input type="number" min={1} value={stepMonths} onChange={(e)=>setStepMonths(parseInt(e.target.value||'1'))} className="w-full px-3 py-2 bg-slate-800/50 border border-white/10 rounded-lg text-white" />
                    </div>
                    <div className="flex items-end">
                      <button onClick={()=>runBacktest(backtestStart, backtestEnd)} disabled={backtestLoading || selectedPortfolio.length===0} className="w-full px-4 py-2 bg-gradient-to-r from-blue-600 to-violet-600 text-white rounded-lg disabled:opacity-50">{backtestLoading ? 'Running...' : 'Run Backtest'}</button>
                    </div>
                  </div>
                  <div className="text-slate-400 text-xs mb-2">Step {currentStep} / {totalSteps}</div>
                  {backtestResults.length > 0 && (
                    <PortfolioChart data={backtestResults.map(r=>({ date: r.date, portfolio_value: r.portfolio_value, spy_value: r.spy_value, alpha: r.alpha }))} />
                  )}
                </GlassCard>
              </motion.div>
            )}

            {/* Analysis */}
            {activeView === 'analysis' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="mb-8 space-y-6"
              >
                <GlassCard className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-white">Portfolio Analysis</h3>
                    <button
                      onClick={async ()=>{
                        const mem = await fetch('http://localhost:8001/api/memory/get', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ thread_id: 'phf' }) });
                        const memData = mem.ok ? await mem.json() : { messages: [] };
                        const payload = { portfolio: selectedPortfolio, backtest_results: backtestResults, memory: memData.messages };
                        const res = await fetch('http://localhost:8001/api/agents/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                        if (res.ok) {
                          const data = await res.json();
                          setSummary(data.summary || '');
                        }
                      }}
                      className="px-4 py-2 bg-slate-800/60 border border-white/10 rounded-lg text-xs text-white"
                    >Summarize Portfolio</button>
                  </div>
                  {selectedPortfolio.length === 0 && (
                    <div className="text-slate-400 text-sm mb-4">No portfolio data. Build a portfolio first.</div>
                  )}
                  {backtestResults.length === 0 && (
                    <div className="text-slate-400 text-sm mb-4">No backtest data. Run a backtest first.</div>
                  )}
                  {summary && (
                    <div className="text-slate-200 text-sm whitespace-pre-wrap">{summary}</div>
                  )}
                </GlassCard>
              </motion.div>
            )}

            {/* Settings */}
            {activeView === 'settings' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="mb-8 space-y-6"
              >
                <GlassCard className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-bold text-white mb-2">Portfolio Settings</h3>
                      <div className="space-y-2">
                        <label className="text-slate-400 text-xs">Max number of stocks</label>
                        <input type="number" min={1} value={target_positions} onChange={(e)=>setTargetPositions(parseInt(e.target.value||'1'))} className="w-full px-3 py-2 bg-slate-800/50 border border-white/10 rounded-lg text-white" />
                      </div>
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-white mb-2">Backtest Settings</h3>
                      <div className="space-y-2">
                        <label className="text-slate-400 text-xs">Rebalance Interval (months)</label>
                        <input type="number" min={1} value={stepMonths} onChange={(e)=>setStepMonths(parseInt(e.target.value||'1'))} className="w-full px-3 py-2 bg-slate-800/50 border border-white/10 rounded-lg text-white" />
                      </div>
                    </div>
                  </div>
                </GlassCard>
              </motion.div>
            )}

            {activeView === 'spoonai' && (
              <HeroMetrics />
            )}

            
          </div>

          {activeView === 'dashboard' && (
            <div className="w-[450px] p-8 pl-0 mt-8">
              <motion.div
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 }}
                className="h-full"
              >
                <AgentStream />
              </motion.div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
