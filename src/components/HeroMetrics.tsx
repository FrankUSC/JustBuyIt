import React from 'react';
import { GlassCard } from './GlassCard';
import { GradientText } from './GradientText';
import { TrendingUp, DollarSign, Activity } from 'lucide-react';
import { useTradingStore } from '../stores/trading';
import { motion } from 'framer-motion';

export const HeroMetrics: React.FC = () => {
  const { totalAssets, alphaGenerated } = useTradingStore();

  const metrics = [
    {
      label: 'Total Assets',
      value: `$${(totalAssets / 1000000).toFixed(2)}M`,
      change: '+12.5%',
      icon: DollarSign,
      positive: true
    },
    {
      label: 'Alpha Generated',
      value: `${alphaGenerated}%`,
      change: '+2.3% vs SPY',
      icon: TrendingUp,
      positive: true
    },
    {
      label: 'Active Positions',
      value: '24',
      change: '8 sectors',
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
                />
              </div>
            </GlassCard>
          </motion.div>
        );
      })}
    </div>
  );
};