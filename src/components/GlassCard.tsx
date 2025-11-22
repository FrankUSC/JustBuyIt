import React from 'react';
import { clsx } from 'clsx';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
}

export const GlassCard: React.FC<GlassCardProps> = ({ children, className }) => {
  return (
    <div className={clsx(
      "relative overflow-hidden",
      "bg-slate-900/60 backdrop-blur-xl",
      "border border-white/10",
      "rounded-3xl",
      "shadow-2xl shadow-black/50",
      "p-6",
      className
    )}>
      {/* Gradient Blob for "Glow" effect */}
      <div className="absolute -top-20 -right-20 w-60 h-60 bg-blue-600/20 rounded-full blur-3xl pointer-events-none" />
      
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};