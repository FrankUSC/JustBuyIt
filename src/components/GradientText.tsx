import React from 'react';

interface GradientTextProps {
  text: string;
  className?: string;
}

export const GradientText: React.FC<GradientTextProps> = ({ text, className }) => {
  return (
    <span className={`bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-indigo-400 to-emerald-400 font-bold ${className || ''}`}>
      {text}
    </span>
  );
};