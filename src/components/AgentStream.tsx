import React, { useEffect, useRef } from 'react';
import { GlassCard } from './GlassCard';
import { useAgentStream } from '../stores/agentStream';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, AlertTriangle, CheckCircle, Info } from 'lucide-react';

export const AgentStream: React.FC = () => {
  const { logs, isScanning } = useAgentStream();
  const terminalRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = terminalRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [logs]);

  const getAgentColor = (agent: string) => {
    switch (agent) {
      case 'SCOUT': return 'text-blue-400';
      case 'EVAL': return 'text-amber-400';
      case 'PORTFOLIO': return 'text-emerald-400';
      case 'BACKTEST': return 'text-violet-400';
      default: return 'text-slate-400';
    }
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'warning': return <AlertTriangle className="w-4 h-4 text-amber-400" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'error': return <Info className="w-4 h-4 text-red-400" />;
      default: return <Activity className="w-4 h-4 text-blue-400" />;
    }
  };

  return (
    <GlassCard className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
        <h3 className="text-xl font-bold text-white">Agent Stream</h3>
        <div className="flex items-center space-x-2">
          {isScanning && (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full"
            />
          )}
          <span className={`text-sm font-medium ${
            isScanning ? 'text-blue-400' : 'text-slate-400'
          }`}>
            {isScanning ? 'Scanning...' : 'Idle'}
          </span>
        </div>
      </div>

      {/* Terminal */}
      <div ref={terminalRef} className="flex-1 min-h-0 overflow-y-auto bg-slate-900/80 rounded-xl p-4 pb-12 font-mono text-sm">
        <div className="space-y-3">
          <AnimatePresence>
            {logs.map((log) => (
              <motion.div
                key={log.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="flex items-start space-x-3 group"
              >
                <div className="flex-shrink-0 mt-0.5">
                  {getIcon(log.type)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2">
                    <span className={`font-bold ${getAgentColor(log.agent)}`}>
                      [{log.agent}]
                    </span>
                    <span className="text-slate-500 text-xs">
                      {log.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-slate-300 mt-1">{log.message}</p>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {logs.length === 0 && (
            <div className="text-center text-slate-500 py-8">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>Agent Stream Ready</p>
              <p className="text-xs mt-1">Waiting for agent activity...</p>
            </div>
          )}
        </div>
      </div>
    </GlassCard>
  );
};