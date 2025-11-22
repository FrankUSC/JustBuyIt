import { create } from 'zustand';

interface TradingState {
  totalAssets: number;
  alphaGenerated: number;
  portfolio: Array<{
    ticker: string;
    shares: number;
    avgPrice: number;
    currentPrice: number;
    value: number;
    weight: number;
  }>;
  isLoading: boolean;
  setTotalAssets: (assets: number) => void;
  setAlphaGenerated: (alpha: number) => void;
  setPortfolio: (portfolio: TradingState['portfolio']) => void;
  setLoading: (loading: boolean) => void;
}

export const useTradingStore = create<TradingState>((set) => ({
  totalAssets: 1250000.00, // $1.25M starting capital
  alphaGenerated: 8.7, // 8.7% alpha vs SPY
  portfolio: [
    { ticker: 'NVDA', shares: 100, avgPrice: 450, currentPrice: 485, value: 48500, weight: 0.10 },
    { ticker: 'AAPL', shares: 200, avgPrice: 180, currentPrice: 185, value: 37000, weight: 0.08 },
    { ticker: 'MSFT', shares: 150, avgPrice: 380, currentPrice: 395, value: 59250, weight: 0.12 },
    { ticker: 'GOOGL', shares: 120, avgPrice: 140, currentPrice: 145, value: 17400, weight: 0.035 }
  ],
  isLoading: false,
  setTotalAssets: (assets) => set({ totalAssets: assets }),
  setAlphaGenerated: (alpha) => set({ alphaGenerated: alpha }),
  setPortfolio: (portfolio) => set({ portfolio }),
  setLoading: (loading) => set({ isLoading: loading })
}));