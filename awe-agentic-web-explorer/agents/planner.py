"""
AWE Planner Agent
=================
Uses Tree of Thought reasoning to plan exploration strategies.

The Planner:
1. Receives observations from the Observer
2. Generates multiple hypotheses about how to achieve the goal
3. Evaluates each hypothesis using ToT
4. Selects the best strategy
5. Creates a concrete action plan
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .base import BaseAgent, AgentCapability
from ..core.types import (
    Action,
    ActionType,
    AgentContext,
    AgentRole,
    ContentLoadingType,
    PageObservation,
    PageType,
    Thought,
    ThoughtTree,
)
from ..core.config import AWEConfig
from ..reasoning.tot import ToTEngine


logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Types
# =============================================================================

@dataclass
class ExplorationStrategy:
    """
    A strategy for exploring a website.
    """
    name: str
    description: str
    actions: List[Action] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""
    
    # For tracking
    selected: bool = False
    executed: bool = False
    success: bool = False


# =============================================================================
# Planner Agent
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    ToT-based strategy planning agent.
    
    Responsibilities:
    - Generate exploration hypotheses
    - Evaluate strategies using ToT
    - Create action plans
    - Handle strategy failures and replanning
    """
    
    # Pre-defined strategy templates
    STRATEGY_TEMPLATES = {
        "ajax_all": ExplorationStrategy(
            name="ajax_all",
            description="Fetch all items via AJAX endpoint with large pagesize",
            actions=[
                Action(type=ActionType.NAVIGATE, target="{ajax_endpoint}?pagesize=500&pageindex=0"),
                Action(type=ActionType.EXTRACT, target="json"),
            ],
            confidence=0.9,
        ),
        "pagination_crawl": ExplorationStrategy(
            name="pagination_crawl",
            description="Crawl through pagination links to collect all items",
            actions=[
                Action(type=ActionType.EXTRACT, target="{item_selector}"),
                Action(type=ActionType.CLICK, target="{next_button}"),
                # Repeat until no more pages
            ],
            confidence=0.7,
        ),
        "scroll_load": ExplorationStrategy(
            name="scroll_load",
            description="Scroll down to trigger lazy loading of more items",
            actions=[
                Action(type=ActionType.SCROLL, target="bottom"),
                Action(type=ActionType.WAIT, value="1000"),
                Action(type=ActionType.EXTRACT, target="{item_selector}"),
                # Repeat until no new items
            ],
            confidence=0.6,
        ),
        "direct_extract": ExplorationStrategy(
            name="direct_extract",
            description="Extract all visible items directly from page",
            actions=[
                Action(type=ActionType.EXTRACT, target="{item_selector}"),
            ],
            confidence=0.5,
        ),
        "search_browse": ExplorationStrategy(
            name="search_browse",
            description="Use search interface to find items, then extract",
            actions=[
                Action(type=ActionType.TYPE, target="{search_input}", value="*"),
                Action(type=ActionType.CLICK, target="{search_button}"),
                Action(type=ActionType.WAIT, value="2000"),
                Action(type=ActionType.EXTRACT, target="{item_selector}"),
            ],
            confidence=0.4,
        ),
    }
    
    def __init__(
        self,
        config: AWEConfig,
        llm_func: Optional[Callable] = None,
        state=None,
    ):
        super().__init__(AgentRole.PLANNER, config, llm_func, state)
        
        # Initialize ToT engine
        self.tot_engine = ToTEngine(llm_func, config) if llm_func else None
        
        # Strategy history
        self._tried_strategies: List[str] = []
        self._failed_strategies: List[str] = []
        
        self.register_capability(AgentCapability(
            name="strategy_planning",
            description="Plan exploration strategies using Tree of Thought reasoning",
            required_inputs=["observation"],
            outputs=["strategy", "action_plan"],
        ))
    
    async def process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Plan an exploration strategy based on current context.
        """
        observation = context.observation
        if not observation:
            return {"error": "No observation available for planning"}
        
        # Generate and select strategy
        strategy = await self.plan(context, observation)
        
        return {
            "strategy": strategy,
            "actions": strategy.actions,
            "confidence": strategy.confidence,
        }
    
    async def plan(
        self,
        context: AgentContext,
        observation: PageObservation,
    ) -> ExplorationStrategy:
        """
        Create an exploration plan based on observation.
        
        Uses ToT if enabled, otherwise falls back to heuristics.
        """
        self.log(f"Planning for {observation.url}", "debug")
        
        # If ToT is enabled, use it for planning
        if self.config.tot_enabled and self.tot_engine:
            return await self._plan_with_tot(context, observation)
        
        # Otherwise use heuristic planning
        return await self._plan_heuristic(observation)
    
    async def _plan_with_tot(
        self,
        context: AgentContext,
        observation: PageObservation,
    ) -> ExplorationStrategy:
        """
        Use Tree of Thought reasoning to plan.
        
        This is where the magic happens - ToT explores multiple
        strategies and picks the best one.
        """
        # Define action executor for ToT
        async def execute_thought(thought: Thought, ctx: AgentContext, obs: PageObservation) -> Dict:
            """Simulate executing a thought to evaluate it."""
            content_lower = thought.content.lower()
            
            # Check if strategy matches page capabilities
            if "ajax" in content_lower:
                if obs.ajax_endpoints:
                    return {
                        "success": True,
                        "data": {
                            "strategy": "ajax_all",
                            "endpoint": obs.ajax_endpoints[0],
                        }
                    }
                return {"success": False, "error": "No AJAX endpoints found"}
            
            if "pagination" in content_lower:
                if obs.pagination_info.get("type"):
                    return {
                        "success": True,
                        "data": {
                            "strategy": "pagination_crawl",
                            "pagination": obs.pagination_info,
                        }
                    }
                return {"success": False, "error": "No pagination found"}
            
            if "scroll" in content_lower:
                if obs.content_loading == ContentLoadingType.AJAX_ON_SCROLL:
                    return {
                        "success": True,
                        "data": {"strategy": "scroll_load"}
                    }
                return {"success": False, "error": "Page doesn't use scroll loading"}
            
            # Default: direct extraction
            if obs.profile_links > 0:
                return {
                    "success": True,
                    "data": {
                        "strategy": "direct_extract",
                        "link_count": obs.profile_links,
                    }
                }
            
            return {"success": False, "error": "No strategy matched"}
        
        self.tot_engine.action_executor = execute_thought
        
        # Run ToT reasoning
        best_thought, result = await self.tot_engine.think(context, observation)
        
        # Convert ToT result to strategy
        strategy = self._thought_to_strategy(best_thought, result, observation)
        
        # Self-reflect if enabled
        if self.config.tot_self_reflection:
            reflection = await self.tot_engine.reflect(context, observation)
            self.log(f"Reflection: {reflection[:100]}", "debug")
        
        return strategy
    
    async def _plan_heuristic(self, observation: PageObservation) -> ExplorationStrategy:
        """
        Heuristic-based planning (no LLM required).
        
        Uses page observation to select the most appropriate strategy.
        """
        strategies = []
        
        # 1. Check for AJAX endpoints (highest priority)
        if observation.ajax_endpoints:
            strategy = ExplorationStrategy(
                name="ajax_all",
                description="Fetch all items via detected AJAX endpoint",
                confidence=0.9,
                reasoning=f"Found AJAX endpoint: {observation.ajax_endpoints[0]}",
            )
            strategy.actions = self._create_ajax_actions(observation)
            strategies.append(strategy)
        
        # 2. Check for pagination
        if observation.pagination_info.get("type"):
            pag = observation.pagination_info
            strategy = ExplorationStrategy(
                name="pagination_crawl",
                description=f"Crawl through {pag['total_pages']} pages",
                confidence=0.7,
                reasoning=f"Found {pag['type']} pagination",
            )
            strategy.actions = self._create_pagination_actions(observation)
            strategies.append(strategy)
        
        # 3. Check for infinite scroll
        if observation.content_loading == ContentLoadingType.AJAX_ON_SCROLL:
            strategy = ExplorationStrategy(
                name="scroll_load",
                description="Scroll to load more items",
                confidence=0.6,
                reasoning="Page uses infinite scroll",
            )
            strategy.actions = self._create_scroll_actions(observation)
            strategies.append(strategy)
        
        # 4. Direct extraction (fallback)
        if observation.profile_links > 0:
            strategy = ExplorationStrategy(
                name="direct_extract",
                description=f"Extract {observation.profile_links} visible items",
                confidence=0.5,
                reasoning="Falling back to visible items",
            )
            strategy.actions = self._create_direct_actions(observation)
            strategies.append(strategy)
        
        if not strategies:
            # No strategy found
            return ExplorationStrategy(
                name="none",
                description="No viable strategy found",
                confidence=0.0,
                reasoning="Page structure not recognized",
            )
        
        # Filter out previously failed strategies
        viable = [s for s in strategies if s.name not in self._failed_strategies]
        if not viable:
            viable = strategies  # Reset if all have failed
        
        # Select best strategy
        best = max(viable, key=lambda s: s.confidence)
        best.selected = True
        
        self.log(f"Selected strategy: {best.name} (confidence: {best.confidence})", "debug")
        
        return best
    
    def _thought_to_strategy(
        self,
        thought: Thought,
        result: Optional[Dict],
        observation: PageObservation,
    ) -> ExplorationStrategy:
        """Convert a ToT thought to a concrete strategy."""
        if not result:
            # Fallback to direct extraction
            return ExplorationStrategy(
                name="direct_extract",
                description="Extract visible items",
                actions=self._create_direct_actions(observation),
                confidence=0.5,
                reasoning=thought.content,
            )
        
        strategy_name = result.get("strategy", "direct_extract")
        
        # Get template and customize
        template = self.STRATEGY_TEMPLATES.get(strategy_name)
        if not template:
            template = self.STRATEGY_TEMPLATES["direct_extract"]
        
        strategy = ExplorationStrategy(
            name=strategy_name,
            description=template.description,
            confidence=thought.confidence,
            reasoning=thought.content,
        )
        
        # Create actions based on strategy type
        if strategy_name == "ajax_all":
            strategy.actions = self._create_ajax_actions(observation, result.get("endpoint"))
        elif strategy_name == "pagination_crawl":
            strategy.actions = self._create_pagination_actions(observation)
        elif strategy_name == "scroll_load":
            strategy.actions = self._create_scroll_actions(observation)
        else:
            strategy.actions = self._create_direct_actions(observation)
        
        return strategy
    
    def _create_ajax_actions(
        self,
        observation: PageObservation,
        endpoint: Optional[str] = None,
    ) -> List[Action]:
        """Create actions for AJAX-based extraction."""
        if not endpoint and observation.ajax_endpoints:
            endpoint = observation.ajax_endpoints[0]
        
        if not endpoint:
            return []
        
        # Build full URL with large pagesize
        from urllib.parse import urljoin
        base_url = observation.url
        full_endpoint = urljoin(base_url, endpoint)
        
        # Add pagesize parameter if not present
        if '?' not in full_endpoint:
            full_endpoint += '?pageindex=0&pagesize=500'
        elif 'pagesize' not in full_endpoint:
            full_endpoint += '&pageindex=0&pagesize=500'
        
        return [
            Action(
                type=ActionType.NAVIGATE,
                target=full_endpoint,
                wait_after=2000,
            ),
            Action(
                type=ActionType.EXTRACT,
                target="json",  # Extract as JSON
            ),
        ]
    
    def _create_pagination_actions(self, observation: PageObservation) -> List[Action]:
        """Create actions for pagination-based crawling."""
        pag = observation.pagination_info
        actions = []
        
        # Extract from current page
        actions.append(Action(
            type=ActionType.EXTRACT,
            target="profile_links",
        ))
        
        # Click next/pagination
        if pag.get("type") == "next_button":
            actions.append(Action(
                type=ActionType.CLICK,
                target=pag.get("selector", 'a:has-text("Next")'),
                wait_after=1000,
            ))
        elif pag.get("type") == "numbered":
            # This would need to be repeated for each page
            for page_num in range(2, pag.get("total_pages", 2) + 1):
                actions.append(Action(
                    type=ActionType.CLICK,
                    target=f'{pag.get("selector", ".pagination")} a:has-text("{page_num}")',
                    wait_after=1000,
                ))
                actions.append(Action(
                    type=ActionType.EXTRACT,
                    target="profile_links",
                ))
        
        return actions
    
    def _create_scroll_actions(self, observation: PageObservation) -> List[Action]:
        """Create actions for scroll-based loading."""
        return [
            Action(type=ActionType.SCROLL, target="bottom", wait_after=1500),
            Action(type=ActionType.EXTRACT, target="profile_links"),
            # Would be repeated until no new items
        ]
    
    def _create_direct_actions(self, observation: PageObservation) -> List[Action]:
        """Create actions for direct extraction."""
        return [
            Action(
                type=ActionType.EXTRACT,
                target="profile_links",
            ),
        ]
    
    def mark_strategy_failed(self, strategy_name: str) -> None:
        """Mark a strategy as failed to avoid retrying."""
        if strategy_name not in self._failed_strategies:
            self._failed_strategies.append(strategy_name)
            self.log(f"Strategy marked as failed: {strategy_name}", "debug")
    
    def reset_failures(self) -> None:
        """Reset failed strategies (for new URLs)."""
        self._failed_strategies.clear()
