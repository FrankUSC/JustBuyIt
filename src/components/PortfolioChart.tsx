import React from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, Legend } from 'recharts';
import { GlassCard } from './GlassCard';
import { TrendingUp } from 'lucide-react';

interface PortfolioChartProps {
  data: Array<{
    date: string;
    portfolio_value: number;
    spy_value: number;
    alpha: number;
  }>;
  height?: number;
}

export const PortfolioChart: React.FC<PortfolioChartProps> = ({ data, height = 400 }) => {
  const chartHeight = Math.max(240, height - 140);
  const formatTooltip = (value: number, name: string) => {
    if (name === 'portfolio_balance') return [`$${Math.round(value).toLocaleString()}`, 'Portfolio'];
    if (name === 'spy_balance') return [`$${Math.round(value).toLocaleString()}`, 'S&P 500'];
    if (name === 'alpha') return [`${value.toFixed(2)}%`, 'Alpha'];
    return [value, name];
  };

  const formatXAxis = (tickItem: string) => {
    return new Date(tickItem).toLocaleDateString('en-US', { month: 'short' });
  };

  const trimmed = React.useMemo(() => {
    if (!data || data.length === 0) return [];
    const last = new Date(data[data.length - 1].date);
    return data.filter(d => {
      const dt = new Date(d.date);
      const diffDays = (last.getTime() - dt.getTime()) / (1000 * 60 * 60 * 24);
      return diffDays <= 365; // past 1 year
    });
  }, [data]);

  const monthly = React.useMemo(() => {
    if (!trimmed.length) return [] as typeof trimmed;
    const lastDate = new Date(trimmed[trimmed.length - 1].date);
    // Build 13 month targets from last month backward
    const monthTargets: Date[] = [];
    for (let i = 12; i >= 0; i--) {
      const dt = new Date(lastDate);
      dt.setMonth(lastDate.getMonth() - i);
      dt.setDate(1);
      dt.setHours(0, 0, 0, 0);
      monthTargets.push(dt);
    }
    // Helper: interpolate value at target date using nearest neighbors
    const interp = (arr: typeof trimmed, key: 'portfolio_value' | 'spy_value', target: Date) => {
      // Find bracketing indices
      let prevIdx = -1;
      let nextIdx = -1;
      for (let i = 0; i < arr.length; i++) {
        const t = new Date(arr[i].date).getTime();
        const tg = target.getTime();
        if (t <= tg) prevIdx = i;
        if (t >= tg) { nextIdx = i; break; }
      }
      if (prevIdx === -1) return arr[0][key];
      if (nextIdx === -1) return arr[arr.length - 1][key];
      if (prevIdx === nextIdx) return arr[prevIdx][key];
      const t0 = new Date(arr[prevIdx].date).getTime();
      const t1 = new Date(arr[nextIdx].date).getTime();
      const v0 = arr[prevIdx][key];
      const v1 = arr[nextIdx][key];
      const alpha = (target.getTime() - t0) / Math.max(1, (t1 - t0));
      return v0 + (v1 - v0) * alpha;
    };
    const out = monthTargets.map(dt => {
      return {
        date: dt.toISOString().split('T')[0],
        portfolio_value: interp(trimmed, 'portfolio_value', dt),
        spy_value: interp(trimmed, 'spy_value', dt),
        alpha: 0,
      };
    });
    return out;
  }, [trimmed]);

  const scaledMonthly = React.useMemo(() => {
    if (!monthly.length) return [] as Array<{
      date: string;
      portfolio_balance: number;
      spy_balance: number;
      alpha: number;
    }>;
    const startBalance = 1000000;
    const firstPortfolio = monthly[0].portfolio_value || 100;
    const firstSpy = monthly[0].spy_value || 100;
    return monthly.map(m => ({
      date: m.date,
      portfolio_balance: Math.round(startBalance * (m.portfolio_value / firstPortfolio)),
      spy_balance: Math.round(startBalance * (m.spy_value / firstSpy)),
      alpha: m.alpha,
    }));
  }, [monthly]);

  const monthTicks = React.useMemo(() => scaledMonthly.map(m => m.date), [scaledMonthly]);

  

  return (
    <GlassCard className={`h-[${height}px]`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-white">Portfolio vs S&P 500</h3>
        <TrendingUp className="w-6 h-6 text-emerald-400" />
      </div>
      
      <div className="h-full flex flex-col">
        <div className="flex-1 h-full min-h-0 min-w-0">
          <ResponsiveContainer width="100%" height={chartHeight} minWidth={0}>
            <AreaChart data={scaledMonthly} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <defs>
                <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="spyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#64748b" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#64748b" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.3} />
              <XAxis 
                dataKey="date" 
                tickFormatter={formatXAxis}
                ticks={monthTicks}
                interval={0}
                minTickGap={0}
                tickMargin={8}
                stroke="#64748b"
                fontSize={12}
              />
              <YAxis 
                stroke="#64748b"
                fontSize={12}
                tickFormatter={(value) => `${Math.round(value).toLocaleString()}`}
              />
              <Tooltip 
                formatter={formatTooltip}
                labelFormatter={(label) => `Date: ${new Date(label).toLocaleDateString()}`}
                contentStyle={{
                  backgroundColor: 'rgba(15, 23, 42, 0.95)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '12px',
                  backdropFilter: 'blur(10px)',
                  color: '#ffffff'
                }}
              />
              <Legend wrapperStyle={{ color: '#cbd5e1' }} />
              <Area
                type="monotone"
                dataKey="spy_balance"
                stroke="#64748b"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#spyGradient)"
                name="S&P 500"
              />
              <Area
                type="monotone"
                dataKey="portfolio_balance"
                stroke="#3b82f6"
                strokeWidth={3}
                fillOpacity={1}
                fill="url(#portfolioGradient)"
                name="Portfolio"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        
        
      </div>
    </GlassCard>
  );
};