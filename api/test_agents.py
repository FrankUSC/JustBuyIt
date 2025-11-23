#!/usr/bin/env python3
"""
Test script for SpoonAI agents
"""

import asyncio
import json
from agents import create_scout_agent, ScoutState

async def test_scout_agent():
    """Test the ScoutAgent"""
    print("🧪 Testing ScoutAgent...")
    
    try:
        scout_agent = create_scout_agent()
        print("✅ ScoutAgent created successfully")
        
        scout_state = ScoutState(
            candidates=[],
            search_criteria={"limit": 5, "min_market_cap": 10000000000},
            stock_count=5,
            natural_query="Find 5 large tech companies with high revenue growth",
        )
        
        print(f"📝 Initial state: {json.dumps(scout_state, indent=2)}")
        
        result = await scout_agent.run(json.dumps(scout_state))
        print(f"📊 Result: {result}")
        
        result_data = json.loads(result)
        print(f"✅ Found {len(result_data.get('candidates', []))} candidates")
        
        if result_data.get('candidates'):
            print("First candidate:", result_data['candidates'][0])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scout_agent())