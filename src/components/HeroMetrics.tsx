import React from 'react';
import { GlassCard } from './GlassCard';
import { GradientText } from './GradientText';
import { TrendingUp, DollarSign, Activity } from 'lucide-react';
import { usePortfolioManager } from '../stores/portfolioManager';
import { useBacktestAgent } from '../stores/backtestAgent';
import { motion } from 'framer-motion';

export const HeroMetrics: React.FC = () => {
  const { portfolio, total_value, cash_remaining } = usePortfolioManager();
  const { results: backtestResults } = useBacktestAgent();

  let endingBalance = (total_value || 0) + (cash_remaining || 0);
  if (backtestResults && backtestResults.length > 1) {
    const first = backtestResults[0];
    const last = backtestResults[backtestResults.length - 1];
    const p0 = first?.portfolio_value || 100;
    const p1 = last?.portfolio_value || p0;
    endingBalance = 1000000 * (p1 / p0);
  }
  const activePositions = portfolio.length;

  let portfolioReturn = 0;
  let spyReturn = 0;
  if (backtestResults && backtestResults.length > 0) {
    const first = backtestResults[0];
    const last = backtestResults[backtestResults.length - 1];
    const p0 = first?.portfolio_value || 100;
    const p1 = last?.portfolio_value || p0;
    const s0 = first?.spy_value || 100;
    const s1 = last?.spy_value || s0;
    portfolioReturn = p0 > 0 ? (p1 / p0 - 1) : 0;
    spyReturn = s0 > 0 ? (s1 / s0 - 1) : 0;
  }
  const alphaGenerated = (portfolioReturn - spyReturn) * 100;

  const metrics = [
    {
      label: 'Total Assets',
      value: `$${(endingBalance / 1000000).toFixed(2)}M`,
      change: `${portfolioReturn >= 0 ? '+' : ''}${(portfolioReturn * 100).toFixed(1)}%`,
      icon: DollarSign,
      positive: portfolioReturn >= 0
    },
    {
      label: 'Alpha Generated',
      value: `${alphaGenerated.toFixed(1)}%`,
      change: `${alphaGenerated >= 0 ? '+' : ''}${alphaGenerated.toFixed(1)}% vs SPY`,
      icon: TrendingUp,
      positive: alphaGenerated >= 0
    },
    {
      label: 'Active Positions',
      value: `${activePositions}`,
      change: `${activePositions} active`,
      icon: Activity,
      positive: true
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        return (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <GlassCard className="p-6 hover:scale-105 transition-transform duration-300">
              <div className="flex items-center justify-between mb-4">
                <div className="p-3 bg-slate-800/50 rounded-xl">
                  <Icon className="w-6 h-6 text-blue-400" />
                </div>
                <span className={`text-sm font-medium ${
                  metric.positive ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {metric.change}
                </span>
              </div>
              
              <div className="space-y-2">
                <p className="text-slate-400 text-sm font-medium">{metric.label}</p>
                <GradientText 
                  text={metric.value} 
                  className="text-4xl font-bold tracking-tight"
                  negative={metric.label === 'Alpha Generated' ? (alphaGenerated < 0) : (metric.label === 'Total Assets' ? (endingBalance < 0) : false)}
                />
              </div>
            </GlassCard>
          </motion.div>
        );
      })}
    </div>
  );
};