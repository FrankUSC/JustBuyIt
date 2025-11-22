import { create } from 'zustand';

interface AgentLog {
  id: string;
  agent: 'SCOUT' | 'EVAL' | 'PORTFOLIO' | 'BACKTEST' | 'SPOONAI' | 'SENTIMENT' | 'RANK';
  message: string;
  timestamp: Date;
  type: 'info' | 'warning' | 'success' | 'error';
}

interface AgentStreamState {
  logs: AgentLog[];
  isScanning: boolean;
  addLog: (agent: AgentLog['agent'], message: string, type?: AgentLog['type']) => void;
  clearLogs: () => void;
  setScanning: (scanning: boolean) => void;
}

export const useAgentStream = create<AgentStreamState>((set) => ({
  logs: [],
  isScanning: false,
  addLog: (agent, message, type = 'info') => {
    const newLog: AgentLog = {
      id: Math.random().toString(36).substr(2, 9),
      agent,
      message,
      timestamp: new Date(),
      type
    };
    set((state) => ({
      logs: [...state.logs, newLog].slice(-100) // Keep last 100 logs
    }));
  },
  clearLogs: () => set({ logs: [] }),
  setScanning: (scanning) => set({ isScanning: scanning })
}));