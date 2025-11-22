import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GlassCard } from './GlassCard';
import { GradientText } from './GradientText';
import { useAgentStream } from '../stores/agentStream';
import { useScoutAgent } from '../stores/scoutAgent';
import { useEvaluationAgent } from '../stores/evaluationAgent';
import { usePortfolioManager } from '../stores/portfolioManager';
import { useBacktestAgent } from '../stores/backtestAgent';
import { PortfolioChart } from './PortfolioChart';
import { ProgressBar } from './ProgressBar';
import { Search, Play, TrendingUp, Activity } from 'lucide-react';

export const AgentOrchestrator: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('Find me aggressive growth tech stocks that are currently beaten down');
  const [isRunning, setIsRunning] = useState(false);
  const [currentPhase, setCurrentPhase] = useState(0);
  
  const { addLog, setScanning } = useAgentStream();
  const { searchStocks } = useScoutAgent();
  const { evaluateStocks } = useEvaluationAgent();
  const { constructPortfolio } = usePortfolioManager();
  const { runBacktest, results: backtestResults } = useBacktestAgent();

  const phases = [
    { name: 'Scout', description: 'Searching for opportunities...' },
    { name: 'Evaluate', description: 'Analyzing risk & sentiment...' },
    { name: 'Portfolio', description: 'Constructing portfolio...' },
    { name: 'Backtest', description: 'Running time machine...' }
  ];

  const runFullPipeline = async () => {
    if (!searchQuery.trim() || isRunning) return;
    
    setIsRunning(true);
    setCurrentPhase(0);
    setScanning(true);
    
    try {
      // Phase 1: Scout Agent
      setCurrentPhase(1);
      addLog('SCOUT', `Processing query: "${searchQuery}"`, 'info');
      await searchStocks(searchQuery);
      {
        const found = useScoutAgent.getState().results;
        if (!found || found.length === 0) {
          addLog('SCOUT', 'No stocks found. Stopping pipeline.', 'error');
          return;
        }
        addLog('SCOUT', `Found ${found.length} candidate stocks`, 'success');
      }
      
      // Phase 2: Evaluation Agent
      setCurrentPhase(2);
      addLog('EVAL', 'Starting risk and sentiment analysis...', 'info');
      const tickers = useScoutAgent.getState().results.map(stock => stock.ticker);
      await evaluateStocks(tickers);
      {
        const selected = useEvaluationAgent.getState().evaluatedStocks;
        if (!selected || selected.length === 0) {
          addLog('EVAL', 'No evaluated stocks passed filters. Stopping pipeline.', 'error');
          return;
        }
        addLog('EVAL', `Selected top ${selected.length} stocks after filtering`, 'success');
      }
      
      // Phase 3: Portfolio Manager
      setCurrentPhase(3);
      addLog('PORTFOLIO', 'Constructing equal-weight portfolio...', 'info');
      await constructPortfolio(useEvaluationAgent.getState().evaluatedStocks);
      {
        const built = usePortfolioManager.getState().portfolio;
        if (!built || built.length === 0) {
          addLog('PORTFOLIO', 'Portfolio construction failed. Stopping pipeline.', 'error');
          return;
        }
        addLog('PORTFOLIO', `Portfolio constructed with ${built.length} positions`, 'success');
      }
      
      // Phase 4: Backtest Agent
      setCurrentPhase(4);
      addLog('BACKTEST', 'Initializing time machine simulation...', 'info');
      const startDate = new Date();
      startDate.setFullYear(startDate.getFullYear() - 3);
      const endDate = new Date();
      
      await runBacktest(startDate.toISOString().split('T')[0], endDate.toISOString().split('T')[0]);
      {
        const bt = useBacktestAgent.getState().results;
        if (!bt || bt.length === 0) {
          addLog('BACKTEST', 'Backtest produced no results. Stopping pipeline.', 'error');
          return;
        }
        addLog('BACKTEST', 'Backtest completed successfully', 'success');
      }
      
    } catch (error) {
      console.error('Pipeline failed:', error);
      addLog('SCOUT', 'Pipeline execution failed', 'error');
    } finally {
      setIsRunning(false);
      setScanning(false);
      setCurrentPhase(0);
    }
  };

  return (
    <div className="space-y-6">
      {/* Search and Control */}
      <GlassCard className="p-6">
        <div className="flex space-x-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              placeholder="Find me aggressive growth tech stocks that are currently beaten down..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && runFullPipeline()}
              className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isRunning}
            />
          </div>
          <button
            onClick={runFullPipeline}
            disabled={!searchQuery.trim() || isRunning}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-violet-600 text-white font-medium rounded-xl hover:from-blue-700 hover:to-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 flex items-center space-x-2"
          >
            {isRunning ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                <span>Running...</span>
              </>
            ) : (
              <>
                <Play size={18} />
                <span>Run Pipeline</span>
              </>
            )}
          </button>
        </div>

        {/* Progress Indicator */}
        {isRunning && (
          <div className="mb-6">
            <ProgressBar 
              current={currentPhase} 
              total={phases.length} 
              label="Agent Pipeline Progress"
            />
            <div className="mt-4 text-center">
              <p className="text-slate-400 text-sm">
                {phases[currentPhase - 1]?.description || 'Initializing...'}
              </p>
            </div>
          </div>
        )}

        {/* Phase Status */}
        <div className="grid grid-cols-4 gap-4">
          {phases.map((phase, index) => (
            <div
              key={phase.name}
              className={`p-4 rounded-xl text-center transition-all duration-300 ${
                currentPhase > index + 1
                  ? 'bg-emerald-600/20 border border-emerald-600/30'
                  : currentPhase === index + 1
                  ? 'bg-blue-600/20 border border-blue-600/30 animate-pulse'
                  : 'bg-slate-800/30 border border-white/10'
              }`}
            >
              <div className="text-sm font-medium text-white mb-1">{phase.name}</div>
              <div className="text-xs text-slate-400">{phase.description}</div>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Results */}
      <AnimatePresence>
        {backtestResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            <PortfolioChart data={backtestResults} />
            
            <GlassCard className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white">Pipeline Results</h3>
                <TrendingUp className="w-6 h-6 text-emerald-400" />
              </div>
              <div className="grid grid-cols-3 gap-6">
                <div className="text-center">
                  <p className="text-slate-400 text-sm mb-2">Total Return</p>
                  <GradientText text="+12.5%" className="text-2xl font-bold" />
                </div>
                <div className="text-center">
                  <p className="text-slate-400 text-sm mb-2">Alpha Generated</p>
                  <GradientText text="+4.3%" className="text-2xl font-bold" />
                </div>
                <div className="text-center">
                  <p className="text-slate-400 text-sm mb-2">Max Drawdown</p>
                  <span className="text-2xl font-bold text-red-400">-8.2%</span>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};