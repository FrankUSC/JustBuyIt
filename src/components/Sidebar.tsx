import React from 'react';
import { BarChart3, TrendingUp, Settings, Activity, Brain } from 'lucide-react';
import { clsx } from 'clsx';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
  { id: 'spoonai', label: 'SpoonAI Agents', icon: Brain },
  { id: 'backtest', label: 'Backtest', icon: TrendingUp },
  { id: 'analysis', label: 'Analysis', icon: Activity },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  return (
    <div className="h-screen w-20 bg-slate-900/80 backdrop-blur-xl border-r border-white/10 flex flex-col items-center py-8 space-y-8">
      {/* Logo */}
      <div className="mb-8">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-violet-600 rounded-xl flex items-center justify-center">
          <span className="text-white font-bold text-lg">PH</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col space-y-6">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={clsx(
                "w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-300 group relative",
                "hover:bg-slate-800/60 hover:scale-110",
                activeView === item.id 
                  ? "bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-500/30"
                  : "bg-slate-800/30 text-slate-400 hover:text-white"
              )}
            >
              <Icon size={24} />
              
              {/* Tooltip */}
              <div className="absolute left-full ml-4 px-3 py-1 bg-slate-800 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-50">
                {item.label}
              </div>
            </button>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="mt-auto">
        <div className="w-14 h-14 rounded-2xl bg-slate-800/30 flex items-center justify-center">
          <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse" />
        </div>
      </div>
    </div>
  );
};