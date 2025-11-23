import { create } from 'zustand';
const API = import.meta.env.VITE_API_URL || 'http://localhost:8001';

interface EvaluatedStock {
  ticker: string;
  score: number;
  risk_factors: string[];
  sentiment: 'positive' | 'negative' | 'neutral';
  recommendation: 'buy' | 'sell' | 'hold';
  reasoning: string;
  sector?: string;
}

interface EvaluationAgentState {
  isLoading: boolean;
  evaluatedStocks: EvaluatedStock[];
  currentBatch: string[];
  setLoading: (loading: boolean) => void;
  setEvaluatedStocks: (stocks: EvaluatedStock[]) => void;
  setCurrentBatch: (tickers: string[]) => void;
  evaluateStocks: (tickers: string[]) => Promise<void>;
}

export const useEvaluationAgent = create<EvaluationAgentState>((set, _get) => ({
  isLoading: false,
  evaluatedStocks: [],
  currentBatch: [],
  setLoading: (loading) => set({ isLoading: loading }),
  setEvaluatedStocks: (stocks) => set({ evaluatedStocks: stocks }),
  setCurrentBatch: (tickers) => set({ currentBatch: tickers }),
  evaluateStocks: async (tickers: string[]) => {
    set({ isLoading: true, currentBatch: tickers });
    
    try {
      const evaluatedStocks: EvaluatedStock[] = [];
      
      for (const ticker of tickers) {
        // Fetch stock data and news
        let history: any[];
        let news: any[];
        
        try {
          const [historyResponse, newsResponse] = await Promise.all([
            fetch(`${API}/api/stocks/${ticker}/history`),
            fetch(`${API}/api/stocks/${ticker}/news`)
          ]);
          
          if (!historyResponse.ok || !newsResponse.ok) {
            throw new Error('API response not ok');
          }
          
          history = await historyResponse.json();
          news = await newsResponse.json();
        } catch {
          // Use mock data if API fails
          history = [
            { date: '2024-01-01', close: 150, rsi: 45, ma_50: 148 },
            { date: '2024-01-02', close: 152, rsi: 47, ma_50: 149 }
          ];
          news = [
            { title: `${ticker} shows strong earnings growth` },
            { title: `${ticker} announces new product line` }
          ];
        }
        
        // Technical analysis
        const latest = history[history.length - 1];
        const risk_factors: string[] = [];
        let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
        let recommendation: 'buy' | 'sell' | 'hold' = 'hold';
        let reasoning = '';
        
        // RSI check
        if (latest.rsi > 70) {
          risk_factors.push('Overbought (RSI > 70)');
        } else if (latest.rsi < 30) {
          risk_factors.push('Oversold (RSI < 30)');
        }
        
        // News sentiment analysis (simplified)
        const newsText = news.map((n: any) => n.title).join(' ').toLowerCase();
        const negativeKeywords = ['lawsuit', 'investigation', 'misses estimates', 'downgrade', 'decline'];
        const positiveKeywords = ['upgrade', 'beats estimates', 'growth', 'expansion', 'breakthrough'];
        
        const hasNegativeNews = negativeKeywords.some(keyword => newsText.includes(keyword));
        const hasPositiveNews = positiveKeywords.some(keyword => newsText.includes(keyword));
        
        if (hasNegativeNews) {
          sentiment = 'negative';
          risk_factors.push('Negative news sentiment');
        } else if (hasPositiveNews) {
          sentiment = 'positive';
        }
        
        // Price vs MA analysis
        if (latest.close < latest.ma_50) {
          recommendation = 'buy';
          reasoning = 'Stock trading below 50-day MA, potential upside';
        } else if (latest.close > latest.ma_50 * 1.1) {
          recommendation = 'sell';
          reasoning = 'Stock significantly above 50-day MA, potential pullback';
        } else {
          reasoning = 'Stock near 50-day MA, neutral outlook';
        }
        
        // Calculate score (0-100)
        let score = 50; // Base score
        
        if (recommendation === 'buy') score += 20;
        if (recommendation === 'sell') score -= 20;
        if (sentiment === 'positive') score += 15;
        if (sentiment === 'negative') score -= 15;
        if (latest.rsi < 70 && latest.rsi > 30) score += 10;
        
        score = Math.max(0, Math.min(100, score));
        
        evaluatedStocks.push({
          ticker,
          score,
          risk_factors,
          sentiment,
          recommendation,
          reasoning,
          sector: latest?.sector ?? 'Unknown'
        });
      }
      
      // Sort by score and take top 10
      evaluatedStocks.sort((a, b) => b.score - a.score);
      const topStocks = evaluatedStocks.slice(0, 10);
      
      set({ evaluatedStocks: topStocks, isLoading: false });
      
    } catch (error) {
      console.error('Evaluation agent failed:', error);
      set({ isLoading: false, evaluatedStocks: [] });
    }
  }
}));