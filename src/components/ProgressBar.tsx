import React from 'react';
import { motion } from 'framer-motion';
import { GlassCard } from './GlassCard';
import { GradientText } from './GradientText';
import { BarChart3, Clock, Target } from 'lucide-react';

interface ProgressBarProps {
  current: number;
  total: number;
  label: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ current, total, label }) => {
  const percentage = total > 0 ? (current / total) * 100 : 0;
  
  return (
    <GlassCard className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-600/20 rounded-lg">
            <BarChart3 className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <p className="text-slate-400 text-sm">{label}</p>
            <p className="text-white font-medium">Step {current} of {total}</p>
          </div>
        </div>
        <div className="text-right">
          <GradientText text={`${percentage.toFixed(1)}%`} className="text-lg font-bold" />
        </div>
      </div>
      
      <div className="relative">
        <div className="w-full bg-slate-800/50 rounded-full h-3 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-600 to-violet-600 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse" />
      </div>
      
      <div className="mt-4 flex items-center justify-between text-sm text-slate-500">
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4" />
          <span>Estimated: ~2 minutes</span>
        </div>
        <div className="flex items-center space-x-2">
          <Target className="w-4 h-4" />
          <span>50 stocks • 10 years</span>
        </div>
      </div>
    </GlassCard>
  );
};