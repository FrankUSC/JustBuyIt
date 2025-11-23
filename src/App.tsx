import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
// import { HeroMetrics } from './components/HeroMetrics';
import { AgentStream } from './components/AgentStream';
import { GlassCard } from './components/GlassCard';
import { GradientText } from './components/GradientText';
import { AgentOrchestrator } from './components/AgentOrchestrator';
import { SpoonAIAgentOrchestrator } from './components/SpoonAIAgentOrchestrator';
import { PortfolioChart } from './components/PortfolioChart';
import { motion } from 'framer-motion';
import { usePortfolioManager } from './stores/portfolioManager';
import { useBacktestAgent } from './stores/backtestAgent';
const API = import.meta.env.VITE_API_URL || 'http://localhost:8001';

function App() {
  const [activeView, setActiveView] = useState('dashboard');
  const { portfolio: selectedPortfolio, setTargetPositions, target_positions } = usePortfolioManager();
  const { results: backtestResults, runBacktest, isLoading: backtestLoading, currentStep, totalSteps, setStepMonths, stepMonths } = useBacktestAgent();
  const [backtestStart, setBacktestStart] = useState('2023-01-01');
  const [backtestEnd, setBacktestEnd] = useState('2025-01-01');
  const [summary, setSummary] = useState<string>('');
  const [llmSummary, setLLMSummary] = useState<string>('');
  const [analystRatings, setAnalystRatings] = useState<Record<string, any>>({});
  const [assetAllocation, setAssetAllocation] = useState<any[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<any>({});
  const [riskPoints, setRiskPoints] = useState<any[]>([]);

  

  

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
                <GradientText text="Just Buy It" />
              </h1>
              <p className="text-slate-400 text-lg">Autonomous AI Quantitative Trading Platform</p>
            </motion.div>

            {/* Dashboard */}
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
                        const mem = await fetch(`${API}/api/memory/get`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ thread_id: 'phf' }) });
                        const memData = mem.ok ? await mem.json() : { messages: [] };
                        const payload = { portfolio: selectedPortfolio, backtest_results: backtestResults, memory: memData.messages, cash: usePortfolioManager.getState().cash_remaining };
                        const res = await fetch(`${API}/api/agents/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                        if (res.ok) {
                          const data = await res.json();
                          setSummary(data.summary || '');
                          // Fallbacks when API returns limited fields
                          const llm = data.llm_summary || `Portfolio of ${selectedPortfolio.length} positions. Top: ${selectedPortfolio.slice(0,5).map(p=>p.ticker).join(', ')}.`;
                          setLLMSummary(llm);
                          const ratings = data.analyst_ratings || Object.fromEntries(selectedPortfolio.map(p=>[p.ticker,{ recommendation: 'neutral', target_mean_price: null, analyst_opinions: null }]));
                          setAnalystRatings(ratings);
                          let allocation = data.asset_allocation || [];
                          if (!allocation.length) {
                            const cashRem = usePortfolioManager.getState().cash_remaining||0;
                            const equities_mv = selectedPortfolio.reduce((s,p)=>s+(p.value||0),0);
                            const total_mv = equities_mv + cashRem;
                            allocation = [
                              { class: 'Equities', market_value: equities_mv, pct_holdings: total_mv>0 ? Math.round((equities_mv/total_mv)*10000)/100 : 0 },
                              { class: 'Cash & Cash Investments', market_value: cashRem, pct_holdings: total_mv>0 ? Math.round((cashRem/total_mv)*10000)/100 : 0 }
                            ];
                          }
                          setAssetAllocation(allocation);
                          let risk = data.risk_metrics || {};
                          let points = data.risk_vs_return_points || [];
                          if (!points.length) {
                            const values = backtestResults.map(r=>r.portfolio_value);
                            const rets: number[] = [];
                            for (let i=1;i<values.length;i++) {
                              const prev = values[i-1] || values[i];
                              rets.push((values[i]-prev)/(prev||values[i]));
                            }
                            if (rets.length) {
                              const mean = rets.reduce((s,x)=>s+x,0)/rets.length;
                              const std = Math.sqrt(rets.reduce((s,x)=>s+(x-mean)*(x-mean),0)/rets.length);
                              const riskPct = Math.round(std*Math.pow(12,0.5)*10000)/100;
                              const retPct = Math.round(mean*12*10000)/100;
                              const rf = 1.0;
                              const sharpe = Math.round(((retPct-rf)/(riskPct||1))*100)/100;
                              risk = { risk_pct: riskPct, return_pct: retPct, sharpe };
                              points = [
                                { label: 'Portfolio', risk: riskPct, return: retPct },
                                { label: 'Aggressive', risk: 15.0, return: 15.3 },
                                { label: 'Moderately Aggressive', risk: 12.85, return: 14.08 },
                                { label: 'Moderate', risk: 9.77, return: 12.35 },
                                { label: 'Moderately Conservative', risk: 6.87, return: 10.10 },
                                { label: 'Conservative', risk: 3.96, return: 8.27 },
                                { label: 'Short Term', risk: 1.83, return: 5.31 },
                                { label: 'Risk Free', risk: 0.21, return: 3.93 }
                              ];
                            }
                          }
                          setRiskMetrics(risk);
                          setRiskPoints(points);
                        } else {
                          console.error('Analysis agent failed:', await res.text());
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
                    <div className="space-y-3">
                      {summary.split('\n').map((line, idx) => {
                        const [label, rest] = line.includes(':') ? [line.split(':')[0], line.split(':').slice(1).join(':').trim()] : [line, ''];
                        if (label.toLowerCase().includes('sector distribution')) {
                          const sectors = rest.split(',').map(s => s.trim());
                          return (
                            <div key={idx} className="text-sm">
                              <span className="text-slate-400">Sector distribution</span>
                              <div className="mt-2 flex flex-wrap gap-2">
                                {sectors.map((sec, i) => (
                                  <span key={i} className="px-2 py-1 bg-slate-800/50 border border-white/10 rounded-md text-slate-200 text-xs">
                                    {sec}
                                  </span>
                                ))}
                              </div>
                            </div>
                          );
                        }
                        return (
                          <div key={idx} className="flex items-center justify-between text-sm">
                            <span className="text-slate-400">{label}</span>
                            {rest && <span className="text-slate-200">{rest}</span>}
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* LLM Narrative */}
                  <div className="mt-4 p-4 bg-slate-800/40 border border-white/10 rounded-xl">
                    <h4 className="text-white font-medium mb-2">LLM Summary</h4>
                    <p className="text-slate-200 text-sm">{llmSummary || 'Summary not available yet.'}</p>
                  </div>

                  {/* Analyst Ratings */}
                  <div className="mt-4 p-4 bg-slate-800/40 border border-white/10 rounded-xl">
                    <h4 className="text-white font-medium mb-2">Analyst Ratings</h4>
                    <div className="text-xs text-slate-300">
                      {selectedPortfolio.map((p) => {
                        const r = (analystRatings as any)[p.ticker] || {};
                        return (
                          <div key={p.ticker} className="flex items-center justify-between py-1">
                            <span className="text-white font-medium">{p.ticker}</span>
                            <span className="text-slate-400">{r.recommendation || 'neutral'}</span>
                            <span className="text-slate-500">TP: {r.target_mean_price ? `$${r.target_mean_price}` : 'n/a'}</span>
                            <span className="text-slate-500">Opinions: {r.analyst_opinions ?? 'n/a'}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Asset Allocation */}
                  <div className="mt-4 p-4 bg-slate-800/40 border border-white/10 rounded-xl">
                    <h4 className="text-white font-medium mb-2">Asset Allocation</h4>
                    <div className="space-y-2">
                      {assetAllocation.length ? assetAllocation.map((a: any) => (
                        <div key={a.class}>
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-slate-300">{a.class}</span>
                            <span className="text-slate-400">${a.market_value.toLocaleString()} · {a.pct_holdings}%</span>
                          </div>
                          <div className="h-2 bg-slate-700/50 rounded">
                            <div className="h-2 bg-blue-600 rounded" style={{ width: `${a.pct_holdings}%` }} />
                          </div>
                        </div>
                      )) : (
                        <div className="text-xs text-slate-400">No allocation data yet.</div>
                      )}
                    </div>
                  </div>

                  {/* Risk vs Return */}
                  <div className="mt-4 p-4 bg-slate-800/40 border border-white/10 rounded-xl">
                    <h4 className="text-white font-medium mb-2">Risk vs Return</h4>
                    <div className="text-xs text-slate-400 mb-2">From {backtestStart} to {backtestEnd}</div>
                    <svg viewBox="0 0 400 250" className="w-full h-64 bg-slate-900/30 rounded">
                      {/* axes */}
                      <line x1="40" y1="210" x2="380" y2="210" stroke="#64748b" strokeWidth="1" />
                      <line x1="40" y1="30" x2="40" y2="210" stroke="#64748b" strokeWidth="1" />
                      {/* points */}
                      {(riskPoints.length ? riskPoints : [{label:'Portfolio',risk:0,return:0}]).map((pt: any, i: number) => {
                        const x = 40 + (pt.risk / 40) * 340; // scale to 0-40
                        const y = 210 - (pt.return / 20) * 180; // scale to 0-20
                        const isPortfolio = pt.label === 'Portfolio';
                        return (
                          <g key={`${pt.label}-${i}`}>
                            <circle cx={x} cy={y} r={isPortfolio ? 6 : 4} fill={isPortfolio ? '#60a5fa' : '#94a3b8'} />
                            <text x={x + 8} y={y - 8} fill="#cbd5e1" fontSize="10">{pt.label}</text>
                          </g>
                        );
                      })}
                    </svg>
                    <div className="mt-2 text-xs text-slate-300">{riskMetrics && riskMetrics.risk_pct !== undefined ? (
                      <>Risk: {riskMetrics.risk_pct}% · Return: {riskMetrics.return_pct}% · Sharpe: {riskMetrics.sharpe}</>
                    ) : (
                      <>Risk: n/a · Return: n/a · Sharpe: n/a</>
                    )}</div>
                  </div>
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

            {activeView === 'spoonai' && null}

            
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
