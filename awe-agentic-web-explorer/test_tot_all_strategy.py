
import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock

# Add current directory to path
cwd = os.getcwd()
sys.path.insert(0, cwd)
print(f"DEBUG: CWD: {cwd}")
print(f"DEBUG: sys.path[0]: {sys.path[0]}")

try:
    import core
    print(f"DEBUG: core imported from: {getattr(core, '__file__', 'unknown')}")
except ImportError as e:
    print(f"DEBUG: Failed to import core: {e}")
    # Try to see what's in CWD
    print(f"DEBUG: Files in CWD: {os.listdir(cwd)}")
    if 'core' in os.listdir(cwd):
        print("DEBUG: 'core' directory exists in CWD")
        if '__init__.py' in os.listdir(os.path.join(cwd, 'core')):
            print("DEBUG: 'core/__init__.py' exists")

try:
    from core.config import AWEConfig
    from core.types import (
        AgentContext, PageObservation, PageType, ContentLoadingType, ExplorationGoal, Thought, ThoughtStatus
    )
    from reasoning.tot import ToTEngine, SearchStrategy
except ImportError as e:
    print(f"CRITICAL ERROR: Import failed: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def mock_llm(prompt: str, **kwargs):
    """Simple mock LLM that returns JSON-like thoughts."""
    if "generate" in prompt.lower():
        # Return valid thoughts with required fields
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
    goal = ExplorationGoal(objective="Find all faculty members", target_fields=[], start_url="http://test.edu/faculty")
    context = MagicMock(spec=AgentContext)
    context.goal = goal
    context.thought_tree = None
    
    observation = PageObservation(
        url="http://test.edu/faculty",
        title="Test Page",
        page_type=PageType.DIRECTORY,
        content_loading=ContentLoadingType.STATIC,
        visible_items=10,
        screenshot_base64="",
        semantic_dom=""
    )
    
    # 4. Run Reasoning
    print(f"Running think() with strategy: {config.tot_search_strategy}")
    best_thought, result = await engine.think(context, observation)
    
    print("\n✅ Test Complete")
    if best_thought:
        print(f"Best Thought: {best_thought.content}")
        print(f"Best Score: {best_thought.score}")
        print(f"Status: {best_thought.status}")
    else:
        print("❌ No best thought returned")

if __name__ == "__main__":
    asyncio.run(run_test())
