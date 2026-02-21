
import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from awe_agentic_web_explorer.core.config import AWEConfig
from awe_agentic_web_explorer.core.types import (
    AgentContext, PageObservation, PageType, ContentLoadingType, Goal, Thought
)
from awe_agentic_web_explorer.reasoning.tot import ToTEngine, SearchStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def mock_llm(prompt: str, **kwargs):
    """Simple mock LLM that returns JSON-like thoughts."""
    if "generate" in prompt.lower():
        return """
        [
            {"content": "Strategy A (BFS like)", "feasibility": 0.8, "confidence": 0.8, "value": 0.8},
            {"content": "Strategy B (DFS like)", "feasibility": 0.7, "confidence": 0.7, "value": 0.7},
            {"content": "Strategy C (Beam like)", "feasibility": 0.6, "confidence": 0.6, "value": 0.6}
        ]
        """
    elif "evaluate" in prompt.lower():
        return '{"should_explore": true, "adjusted_feasibility": 0.9, "adjusted_confidence": 0.9, "adjusted_value": 0.9}'
    elif "reflect" in prompt.lower():
        return "Reflection: Good job."
    return ""

async def mock_executor(thought, context, observation):
    """Mock execution that always succeeds."""
    return {"success": True, "data": f"Executed {thought.content}"}

async def run_test():
    print("🚀 Starting ToT ALL Strategy Test")
    
    # 1. Setup Config
    config = AWEConfig()
    config.tot_enabled = True
    config.tot_search_strategy = "all"  # Testing the new ALL strategy
    config.tot_max_thoughts = 3
    config.tot_max_depth = 2
    
    # 2. Setup Engine
    engine = ToTEngine(mock_llm, config, action_executor=mock_executor)
    
    # 3. Setup Mock Data
    goal = Goal(objective="Find all faculty members")
    context = MagicMock(spec=AgentContext)
    context.goal = goal
    context.thought_tree = None
    
    observation = PageObservation(
        url="http://test.edu/faculty",
        page_type=PageType.DIRECTORY,
        content_loading=ContentLoadingType.STATIC,
        visible_items=10,
        screenshot="",
        dom_tree=""
    )
    
    # 4. Run Reasoning
    print(f"Running think() with strategy: {config.tot_search_strategy}")
    best_thought, result = await engine.think(context, observation)
    
    print("\n✅ Test Complete")
    print(f"Best Thought: {best_thought.content}")
    print(f"Best Score: {best_thought.score}")
    print(f"Status: {best_thought.status}")

if __name__ == "__main__":
    asyncio.run(run_test())
