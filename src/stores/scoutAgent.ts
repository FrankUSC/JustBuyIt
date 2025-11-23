import { create } from 'zustand';
const API = import.meta.env.VITE_API_URL || 'http://localhost:8001';

interface Stock {
  ticker: string;
  sector: string;
  market_cap: number;
  revenue_growth: number;
  current_price: number;
  current_rsi: number;
  ma_50: number;
}

interface ScoutAgentState {
  isLoading: boolean;
  results: Stock[];
  query: string;
  setLoading: (loading: boolean) => void;
  setResults: (results: Stock[]) => void;
  setQuery: (query: string) => void;
  searchStocks: (query: string) => Promise<void>;
}

export const useScoutAgent = create<ScoutAgentState>((set, _get) => ({
  isLoading: false,
  results: [],
  query: '',
  setLoading: (loading) => set({ isLoading: loading }),
  setResults: (results) => set({ results }),
  setQuery: (query) => set({ query }),
  searchStocks: async (query: string) => {
    set({ isLoading: true, query });
    
    try {
      const response = await fetch(`${API}/api/agents/scout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, stock_count: 30 })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const payload = await response.json();
      const candidates = Array.isArray(payload) ? payload : (payload.candidates || []);
      set({ results: candidates, isLoading: false });
      
    } catch (error) {
      console.error('Scout agent search failed:', error);
      
      // Fallback to mock data if API fails
      const mockResults = [
        {
          ticker: 'NVDA',
          sector: 'Technology',
          market_cap: 1200000000000,
          revenue_growth: 22.4,
          current_price: 485.50,
          current_rsi: 45.2,
          ma_50: 472.30
        },
        {
          ticker: 'AAPL',
          sector: 'Technology',
          market_cap: 3000000000000,
          revenue_growth: 8.2,
          current_price: 185.25,
          current_rsi: 52.1,
          ma_50: 180.15
        },
        {
          ticker: 'MSFT',
          sector: 'Technology',
          market_cap: 2800000000000,
          revenue_growth: 12.1,
          current_price: 395.80,
          current_rsi: 48.7,
          ma_50: 388.90
        }
      ];
      
      set({ results: mockResults, isLoading: false });
    }
  }
}));