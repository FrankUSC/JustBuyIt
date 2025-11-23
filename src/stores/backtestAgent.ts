import { create } from 'zustand';
import { usePortfolioManager } from './portfolioManager';

interface BacktestResult {
  date: string;
  portfolio_value: number;
  spy_value: number;
  alpha: number;
  positions: Array<{
    ticker: string;
    shares: number;
    price: number;
    value: number;
  }>;
}

interface BacktestAgentState {
  isLoading: boolean;
  results: BacktestResult[];
  currentStep: number;
  totalSteps: number;
  startDate: string;
  endDate: string;
  runBacktest: (startDate: string, endDate: string) => Promise<void>;
  stepMonths: number;
  setStepMonths: (m: number) => void;
}

const DEFAULT_STEP_MONTHS = 1;

export const useBacktestAgent = create<BacktestAgentState>((set, get) => ({
  isLoading: false,
  results: [],
  currentStep: 0,
  totalSteps: 0,
  startDate: '',
  endDate: '',
  stepMonths: DEFAULT_STEP_MONTHS,
  setStepMonths: (m) => set({ stepMonths: Math.max(1, Math.floor(m)) }),
  
  runBacktest: async (startDate: string, endDate: string) => {
    set({ isLoading: true, startDate, endDate, currentStep: 0, results: [] });
    
    try {
      const start = new Date(startDate);
      const end = new Date(endDate);
      const totalMonths = (end.getFullYear() - start.getFullYear()) * 12 + (end.getMonth() - start.getMonth());
      const stepMonths = get().stepMonths;
      const totalSteps = Math.ceil(totalMonths / stepMonths) + 1;
      // console.log("totalMonths", totalMonths);
      // console.log("totalSteps", totalSteps);  
      // console.log("stepMonths", stepMonths);

      set({ totalSteps });
      
      const results: BacktestResult[] = [];
      let _currentCapital = 1000000;
      let spyValue = 100;
      let portfolioValue = 100;

      const holdings = usePortfolioManager.getState().portfolio;
      if (!holdings || holdings.length === 0) {
        set({ isLoading: false });
        return;
      }

      const fetchHistory = async (ticker: string) => {
        const res = await fetch(`http://localhost:8001/api/stocks/${ticker}/history?period=1y`);
        if (!res.ok) return [] as Array<{ date: string; close: number }>;
        const data = await res.json();
        return (data as Array<any>).map(d => ({ date: d.date, close: d.close }));
      };

      const spyHistory = await fetchHistory('SPY');
      const holdingsHistory: Record<string, Array<{ date: string; close: number }>> = {};
      for (const h of holdings) {
        holdingsHistory[h.ticker] = await fetchHistory(h.ticker);
      }

      const findCloseOnOrBefore = (hist: Array<{ date: string; close: number }>, d: Date) => {
        for (let i = hist.length - 1; i >= 0; i--) {
          const hd = new Date(hist[i].date);
          if (hd.getTime() <= d.getTime()) return hist[i].close;
        }
        return hist.length ? hist[0].close : 0;
      };

      let prevDate = new Date(start);
      for (let step = 0; step < totalSteps; step++) {
        const stepDate = new Date(start);
        stepDate.setMonth(start.getMonth() + (step * stepMonths));
        set({ currentStep: step + 1 });

        const spyStart = findCloseOnOrBefore(spyHistory, prevDate);
        const spyEnd = findCloseOnOrBefore(spyHistory, stepDate);
        const marketReturn = spyStart > 0 ? (spyEnd / spyStart - 1) : 0;

        let weightedReturn = 0;
        const equalWeight = 1 / holdings.length;
        for (const h of holdings) {
          const hh = holdingsHistory[h.ticker] || [];
          const ps = findCloseOnOrBefore(hh, prevDate);
          const pe = findCloseOnOrBefore(hh, stepDate);
          const r = ps > 0 ? (pe / ps - 1) : 0;
          weightedReturn += r * equalWeight;
        }
        const portfolioReturn = weightedReturn;
        const alpha = portfolioReturn - marketReturn;

        portfolioValue *= (1 + portfolioReturn);
        spyValue *= (1 + marketReturn);
        _currentCapital *= (1 + portfolioReturn);

        const positions = holdings.map(h => {
          const price = findCloseOnOrBefore(holdingsHistory[h.ticker] || [], stepDate);
          const value = h.shares * price;
          return { ticker: h.ticker, shares: h.shares, price, value };
        });

        results.push({
          date: stepDate.toISOString().split('T')[0],
          portfolio_value: portfolioValue,
          spy_value: spyValue,
          alpha: alpha * 100,
          positions
        });

        prevDate = new Date(stepDate);
      }
      
      set({ 
        results, 
        currentStep: totalSteps 
      });
      try {
        await fetch('http://localhost:8001/api/memory/append', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ thread_id: 'phf', role: 'assistant', content: `Backtest completed ${totalSteps} steps`, meta: { type: 'backtest', startDate, endDate, steps: totalSteps } })
        });
      } catch (e) { void e }
      set({ isLoading: false });
      
    } catch (error) {
      console.error('Backtest failed:', error);
      set({ isLoading: false });
    }
  }
}));