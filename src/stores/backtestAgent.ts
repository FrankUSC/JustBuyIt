import { create } from 'zustand';

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

const DEFAULT_STEP_MONTHS = 6;

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
      const totalSteps = Math.ceil(totalMonths / stepMonths);
      
      set({ totalSteps });
      
      const results: BacktestResult[] = [];
      let currentCapital = 1250000; // Starting with $1.25M
      let spyValue = 100; // Normalized SPY starting value
      let portfolioValue = 100; // Normalized portfolio starting value
      
      // Simulate backtest steps
      for (let step = 0; step < totalSteps; step++) {
        const stepDate = new Date(start);
        stepDate.setMonth(start.getMonth() + (step * stepMonths));
        
        set({ currentStep: step + 1 });
        
        // Simulate market movements and portfolio performance
        const marketReturn = (Math.random() - 0.5) * 0.2; // ±10% per 6 months
        const alpha = (Math.random() - 0.3) * 0.1; // Generate alpha (slightly positive bias)
        const portfolioReturn = marketReturn + alpha;
        
        portfolioValue *= (1 + portfolioReturn);
        spyValue *= (1 + marketReturn);
        currentCapital *= (1 + portfolioReturn);
        
        // Simulate positions (simplified)
        const positions = [
          { ticker: 'NVDA', shares: Math.floor(Math.random() * 100) + 50, price: 400 + Math.random() * 100, value: 0 },
          { ticker: 'AAPL', shares: Math.floor(Math.random() * 200) + 100, price: 150 + Math.random() * 50, value: 0 },
          { ticker: 'MSFT', shares: Math.floor(Math.random() * 150) + 75, price: 300 + Math.random() * 100, value: 0 }
        ];
        
        // Calculate position values
        positions.forEach(pos => {
          pos.value = pos.shares * pos.price;
        });
        
        results.push({
          date: stepDate.toISOString().split('T')[0],
          portfolio_value: portfolioValue,
          spy_value: spyValue,
          alpha: alpha * 100, // Convert to percentage
          positions
        });
        
        // Add small delay to simulate processing
        await new Promise(resolve => setTimeout(resolve, 500));
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