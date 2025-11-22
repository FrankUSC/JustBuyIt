import { create } from 'zustand';

interface PortfolioStock {
  ticker: string;
  shares: number;
  avg_price: number;
  current_price: number;
  value: number;
  weight: number;
  score: number;
}

interface PortfolioManagerState {
  isLoading: boolean;
  portfolio: PortfolioStock[];
  total_value: number;
  cash_remaining: number;
  target_positions: number;
  constructPortfolio: (evaluatedStocks: any[]) => Promise<void>;
  rebalancePortfolio: () => Promise<void>;
  setTargetPositions: (n: number) => void;
}

const TOTAL_CAPITAL = 1250000;

export const usePortfolioManager = create<PortfolioManagerState>((set, get) => ({
  isLoading: false,
  portfolio: [],
  total_value: 0,
  cash_remaining: TOTAL_CAPITAL,
  target_positions: 10,
  setTargetPositions: (n) => set({ target_positions: Math.max(1, Math.floor(n)) }),
  
  constructPortfolio: async (evaluatedStocks: any[]) => {
    set({ isLoading: true });
    
    try {
      const { target_positions } = get();
      const topStocks = evaluatedStocks.slice(0, target_positions);
      const equal_weight = topStocks.length > 0 ? 1.0 / topStocks.length : 0;
      const position_value = TOTAL_CAPITAL * equal_weight;
      
      const portfolio: PortfolioStock[] = topStocks.map((stock: any) => {
        const shares = Math.floor(position_value / stock.current_price);
        const value = shares * stock.current_price;
        
        return {
          ticker: stock.ticker,
          shares,
          avg_price: stock.current_price,
          current_price: stock.current_price,
          value,
          weight: equal_weight,
          score: stock.score
        };
      });
      
      const total_invested = portfolio.reduce((sum, pos) => sum + pos.value, 0);
      const cash_remaining = TOTAL_CAPITAL - total_invested;
      
      set({ 
        portfolio, 
        total_value: total_invested,
        cash_remaining,
        isLoading: false 
      });
      
    } catch (error) {
      console.error('Portfolio construction failed:', error);
      set({ isLoading: false });
    }
  },
  
  rebalancePortfolio: async () => {
    set({ isLoading: true });
    
    try {
      const { portfolio, total_value } = get();
      const current_total = portfolio.reduce((sum, pos) => sum + (pos.shares * pos.current_price), 0);
      const equal_weight = 1.0 / portfolio.length;
      
      const rebalancedPortfolio = portfolio.map(stock => {
        const current_value = stock.shares * stock.current_price;
        const target_value = current_total * equal_weight;
        const target_shares = Math.floor(target_value / stock.current_price);
        
        return {
          ...stock,
          shares: target_shares,
          value: target_shares * stock.current_price,
          weight: equal_weight
        };
      });
      
      set({ 
        portfolio: rebalancedPortfolio,
        total_value: current_total,
        isLoading: false 
      });
      
    } catch (error) {
      console.error('Portfolio rebalancing failed:', error);
      set({ isLoading: false });
    }
  }
}));