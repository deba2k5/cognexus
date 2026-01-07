"""
AWE Tree of Thought (ToT) Reasoning Engine
==========================================
Implements deliberate problem solving through multi-path exploration.

Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
(Yao et al., 2023) - https://arxiv.org/abs/2305.10601

Key insight: ToT enables smaller models to achieve reasoning capabilities
comparable to larger models by exploring multiple paths and self-evaluating.
"""

from __future__ import annotations
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

from ..core.types import (
    Thought,
    ThoughtTree,
    ThoughtStatus,
    PageObservation,
    AgentContext,
)
from ..core.config import AWEConfig


logger = logging.getLogger(__name__)


# =============================================================================
# ToT Search Strategies
# =============================================================================

class SearchStrategy(str, Enum):
    """Search strategies for exploring the thought tree."""
    BFS = "bfs"      # Breadth-first: explore all thoughts at depth d before d+1
    DFS = "dfs"      # Depth-first: explore one path fully, then backtrack
    BEAM = "beam"    # Beam search: keep top-k paths at each depth


# =============================================================================
# Thought Generator
# =============================================================================

class ThoughtGenerator:
    """
    Generates candidate thoughts (approaches) for a given context.
    
    Uses the LLM to brainstorm multiple ways to solve the current problem,
    considering the observation, goal, and history.
    """
    
    GENERATION_PROMPT = """You are an intelligent web exploration agent. Given the current context,
generate {n} distinct approaches to achieve the goal.

GOAL: {goal}

CURRENT OBSERVATION:
- Page URL: {url}
- Page Type: {page_type}
- Content Loading: {content_loading}
- Visible Items: {visible_items}
- Key Patterns Found: {patterns}

PREVIOUS ATTEMPTS (if any):
{history}

Generate {n} different approaches. For each approach, provide:
1. A clear description of what to try
2. Why this might work (reasoning)
3. Estimated feasibility (0.0-1.0)
4. Estimated confidence of success (0.0-1.0)
5. Estimated value/importance (0.0-1.0)

Think step by step. Consider:
- AJAX/API endpoints that might return all data
- Pagination patterns (URL, click, scroll)
- CSS selectors for extracting data
- Alternative navigation paths

Output as JSON array:
[
  {{
    "content": "Description of approach",
    "reasoning": "Why this might work",
    "feasibility": 0.8,
    "confidence": 0.7,
    "value": 0.9
  }},
  ...
]
"""
    
    def __init__(self, llm_func: Callable, config: AWEConfig):
        """
        Args:
            llm_func: Async function that takes a prompt and returns text
            config: AWE configuration
        """
        self.llm = llm_func
        self.config = config
    
    async def generate(
        self,
        context: AgentContext,
        observation: PageObservation,
        n: int = None,
        parent_thought: Optional[Thought] = None,
    ) -> List[Thought]:
        """
        Generate n candidate thoughts for the current context.
        
        Args:
            context: Current agent context
            observation: Current page observation
            n: Number of thoughts to generate (uses config default if None)
            parent_thought: Parent thought if generating children
        
        Returns:
            List of generated Thought objects
        """
        n = n or self.config.tot_max_thoughts
        
        # Build history from parent chain
        history = ""
        if parent_thought:
            history = self._build_history(context.thought_tree, parent_thought)
        
        # Build prompt
        prompt = self.GENERATION_PROMPT.format(
            n=n,
            goal=context.goal.to_prompt(),
            url=observation.url,
            page_type=observation.page_type.value,
            content_loading=observation.content_loading.value,
            visible_items=observation.visible_items,
            patterns=", ".join(observation.link_patterns[:5]) or "None detected",
            history=history or "No previous attempts",
        )
        
        # Call LLM
        try:
            response = await self.llm(prompt)
            thoughts = self._parse_thoughts(response, parent_thought)
            logger.debug(f"Generated {len(thoughts)} thoughts")
            return thoughts
        except Exception as e:
            logger.error(f"Thought generation failed: {e}")
            # Return a default thought
            return [Thought(
                id=str(uuid.uuid4()),
                content="Extract visible items using detected selectors",
                parent_id=parent_thought.id if parent_thought else None,
                depth=parent_thought.depth + 1 if parent_thought else 0,
                feasibility=0.5,
                confidence=0.5,
                value=0.5,
            )]
    
    def _build_history(self, tree: Optional[ThoughtTree], current: Thought) -> str:
        """Build history string from thought chain."""
        if not tree:
            return ""
        
        history = []
        thought = current
        while thought.parent_id:
            parent = tree.thoughts.get(thought.parent_id)
            if parent:
                status = "✅ succeeded" if parent.status == ThoughtStatus.SUCCEEDED else "❌ failed"
                history.append(f"- {parent.content} [{status}]")
                if parent.result:
                    history.append(f"  Result: {parent.result[:100]}")
                thought = parent
            else:
                break
        
        return "\n".join(reversed(history))
    
    def _parse_thoughts(
        self,
        response: str,
        parent: Optional[Thought],
    ) -> List[Thought]:
        """Parse LLM response into Thought objects."""
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.warning("No JSON found in thought generation response")
            return []
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse thought JSON")
            return []
        
        thoughts = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            
            thoughts.append(Thought(
                id=str(uuid.uuid4()),
                content=item.get("content", f"Approach {i+1}"),
                parent_id=parent.id if parent else None,
                depth=parent.depth + 1 if parent else 0,
                feasibility=float(item.get("feasibility", 0.5)),
                confidence=float(item.get("confidence", 0.5)),
                value=float(item.get("value", 0.5)),
            ))
        
        return thoughts


# =============================================================================
# Thought Evaluator
# =============================================================================

class ThoughtEvaluator:
    """
    Evaluates and scores thoughts based on feasibility, potential, and context.
    
    Uses a combination of:
    1. LLM-based evaluation (asks LLM to score the thought)
    2. Heuristic rules (pattern matching, history analysis)
    3. Past success/failure memory
    """
    
    EVALUATION_PROMPT = """You are evaluating whether to explore a potential approach for web exploration.

GOAL: {goal}

CURRENT APPROACH TO EVALUATE:
{thought_content}

CONTEXT:
- Page URL: {url}
- Page Type: {page_type}
- Available patterns: {patterns}

Previous outcomes at this step:
{sibling_outcomes}

Should we explore this approach? Consider:
1. Is it technically feasible given the page structure?
2. How likely is it to succeed?
3. Does it offer value beyond other approaches?
4. Have similar approaches failed before?

Output JSON:
{{
  "should_explore": true/false,
  "adjusted_feasibility": 0.0-1.0,
  "adjusted_confidence": 0.0-1.0,
  "adjusted_value": 0.0-1.0,
  "reasoning": "Brief explanation"
}}
"""
    
    def __init__(self, llm_func: Callable, config: AWEConfig):
        self.llm = llm_func
        self.config = config
        
        # Memory of past evaluations for similar thoughts
        self._history: Dict[str, float] = {}
    
    async def evaluate(
        self,
        thought: Thought,
        context: AgentContext,
        observation: PageObservation,
        use_llm: bool = True,
    ) -> Thought:
        """
        Evaluate a thought and update its scores.
        
        Args:
            thought: The thought to evaluate
            context: Current agent context
            observation: Current page observation
            use_llm: Whether to use LLM for evaluation
        
        Returns:
            Updated Thought with adjusted scores
        """
        # Apply heuristic adjustments first
        thought = self._apply_heuristics(thought, observation)
        
        # Check history for similar thoughts
        thought = self._check_history(thought)
        
        # Use LLM for complex evaluation if enabled
        if use_llm and self.config.tot_enabled:
            thought = await self._llm_evaluate(thought, context, observation)
        
        return thought
    
    def _apply_heuristics(self, thought: Thought, observation: PageObservation) -> Thought:
        """Apply heuristic rules to adjust thought scores."""
        content_lower = thought.content.lower()
        
        # Boost API/AJAX approaches if endpoints detected
        if observation.ajax_endpoints and ("ajax" in content_lower or "api" in content_lower):
            thought.confidence = min(1.0, thought.confidence + 0.2)
            thought.value = min(1.0, thought.value + 0.1)
        
        # Boost pagination approaches if pagination detected
        if observation.pagination_info and "pagination" in content_lower:
            thought.confidence = min(1.0, thought.confidence + 0.15)
        
        # Penalize scroll-based if no infinite scroll detected
        if "scroll" in content_lower and observation.content_loading.value != "ajax_on_scroll":
            thought.confidence = max(0.1, thought.confidence - 0.2)
        
        # Boost selector-based if specific selectors mentioned
        if "selector" in content_lower or "css" in content_lower:
            if observation.card_patterns:
                thought.feasibility = min(1.0, thought.feasibility + 0.1)
        
        return thought
    
    def _check_history(self, thought: Thought) -> Thought:
        """Check if similar thoughts have succeeded/failed before."""
        # Simple keyword matching for history lookup
        keywords = set(thought.content.lower().split())
        
        for past_content, past_score in self._history.items():
            past_keywords = set(past_content.lower().split())
            overlap = len(keywords & past_keywords) / max(len(keywords), 1)
            
            if overlap > 0.5:  # High similarity
                # Adjust based on past success
                if past_score > 0.7:
                    thought.confidence = min(1.0, thought.confidence + 0.1)
                elif past_score < 0.3:
                    thought.confidence = max(0.1, thought.confidence - 0.2)
                break
        
        return thought
    
    async def _llm_evaluate(
        self,
        thought: Thought,
        context: AgentContext,
        observation: PageObservation,
    ) -> Thought:
        """Use LLM for detailed evaluation."""
        # Get sibling outcomes for context
        sibling_outcomes = ""
        if context.thought_tree and thought.parent_id:
            siblings = context.thought_tree.get_children(thought.parent_id)
            for sib in siblings:
                if sib.id != thought.id and sib.status != ThoughtStatus.PENDING:
                    sibling_outcomes += f"- {sib.content[:50]}: {sib.status.value}\n"
        
        prompt = self.EVALUATION_PROMPT.format(
            goal=context.goal.objective,
            thought_content=thought.content,
            url=observation.url,
            page_type=observation.page_type.value,
            patterns=", ".join(observation.link_patterns[:3]) or "None",
            sibling_outcomes=sibling_outcomes or "None yet",
        )
        
        try:
            response = await self.llm(prompt)
            self._parse_evaluation(thought, response)
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
        
        return thought
    
    def _parse_evaluation(self, thought: Thought, response: str) -> None:
        """Parse LLM evaluation response and update thought."""
        import json
        import re
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return
        
        try:
            data = json.loads(json_match.group())
            
            if "adjusted_feasibility" in data:
                thought.feasibility = float(data["adjusted_feasibility"])
            if "adjusted_confidence" in data:
                thought.confidence = float(data["adjusted_confidence"])
            if "adjusted_value" in data:
                thought.value = float(data["adjusted_value"])
            
        except (json.JSONDecodeError, ValueError):
            pass
    
    def record_outcome(self, thought: Thought) -> None:
        """Record the outcome of a thought for future reference."""
        score = 1.0 if thought.status == ThoughtStatus.SUCCEEDED else 0.0
        self._history[thought.content] = score
    
    async def evaluate_batch(
        self,
        thoughts: List[Thought],
        context: AgentContext,
        observation: PageObservation,
    ) -> List[Thought]:
        """Evaluate multiple thoughts in parallel."""
        tasks = [
            self.evaluate(thought, context, observation)
            for thought in thoughts
        ]
        return await asyncio.gather(*tasks)


# =============================================================================
# ToT Search Engine
# =============================================================================

class ToTEngine:
    """
    Main Tree of Thought reasoning engine.
    
    Coordinates:
    1. Thought generation
    2. Thought evaluation
    3. Tree exploration (BFS/DFS/Beam)
    4. Backtracking on failure
    5. Self-reflection and learning
    """
    
    def __init__(
        self,
        llm_func: Callable,
        config: AWEConfig,
        action_executor: Optional[Callable] = None,
    ):
        """
        Args:
            llm_func: Async function for LLM calls
            config: AWE configuration
            action_executor: Optional function to execute thoughts and return results
        """
        self.llm = llm_func
        self.config = config
        self.action_executor = action_executor
        
        self.generator = ThoughtGenerator(llm_func, config)
        self.evaluator = ThoughtEvaluator(llm_func, config)
        
        self.tree = ThoughtTree()
        self._best_result: Optional[Any] = None
        self._exploration_count = 0
    
    async def think(
        self,
        context: AgentContext,
        observation: PageObservation,
    ) -> Tuple[Thought, Any]:
        """
        Main entry point for ToT reasoning.
        
        Generates, evaluates, and explores thoughts to find the best approach.
        
        Args:
            context: Current agent context
            observation: Current page observation
        
        Returns:
            Tuple of (best thought, result from executing that thought)
        """
        self.tree = ThoughtTree()
        self._best_result = None
        self._exploration_count = 0
        
        logger.info(f"Starting ToT reasoning with strategy: {self.config.tot_search_strategy}")
        
        # Generate initial thoughts
        initial_thoughts = await self.generator.generate(context, observation)
        
        # Evaluate initial thoughts
        evaluated = await self.evaluator.evaluate_batch(initial_thoughts, context, observation)
        
        # Add to tree
        for thought in evaluated:
            self.tree.add_thought(thought)
        
        # Explore based on strategy
        if self.config.tot_search_strategy == SearchStrategy.BFS.value:
            best = await self._explore_bfs(context, observation)
        elif self.config.tot_search_strategy == SearchStrategy.DFS.value:
            best = await self._explore_dfs(context, observation)
        else:  # beam
            best = await self._explore_beam(context, observation)
        
        logger.info(f"ToT complete. Explored {self._exploration_count} thoughts. Best score: {best.score:.2f}")
        
        return best, self._best_result
    
    async def _explore_bfs(
        self,
        context: AgentContext,
        observation: PageObservation,
    ) -> Thought:
        """Breadth-first exploration of thoughts."""
        current_depth = 0
        
        while current_depth < self.config.tot_max_depth:
            # Get all thoughts at current depth
            thoughts_at_depth = [
                t for t in self.tree.thoughts.values()
                if t.depth == current_depth and t.status == ThoughtStatus.PENDING
            ]
            
            if not thoughts_at_depth:
                break
            
            # Sort by score
            thoughts_at_depth.sort(key=lambda t: t.score, reverse=True)
            
            # Explore each thought
            for thought in thoughts_at_depth:
                if thought.score < self.config.tot_min_score:
                    thought.status = ThoughtStatus.ABANDONED
                    continue
                
                result = await self._explore_thought(thought, context, observation)
                
                if thought.status == ThoughtStatus.SUCCEEDED:
                    # Found a working approach
                    return thought
            
            current_depth += 1
        
        # Return best thought overall
        return self._get_best_thought()
    
    async def _explore_dfs(
        self,
        context: AgentContext,
        observation: PageObservation,
    ) -> Thought:
        """Depth-first exploration with backtracking."""
        # Get highest-scoring root thought
        root_thoughts = [self.tree.thoughts[rid] for rid in self.tree.root_ids]
        if not root_thoughts:
            raise ValueError("No thoughts to explore")
        
        current = max(root_thoughts, key=lambda t: t.score)
        self.tree.current_path = [current.id]
        
        while current:
            if current.depth >= self.config.tot_max_depth:
                # Max depth reached, try to backtrack
                current = self.tree.backtrack()
                continue
            
            result = await self._explore_thought(current, context, observation)
            
            if current.status == ThoughtStatus.SUCCEEDED:
                return current
            
            if current.status == ThoughtStatus.FAILED:
                # Backtrack
                current = self.tree.backtrack()
                continue
            
            # Generate children for current thought
            children = await self.generator.generate(
                context, observation, parent_thought=current
            )
            
            if not children:
                current.status = ThoughtStatus.FAILED
                current = self.tree.backtrack()
                continue
            
            # Evaluate and add children
            children = await self.evaluator.evaluate_batch(children, context, observation)
            for child in children:
                self.tree.add_thought(child)
            
            # Move to best child
            current = max(children, key=lambda t: t.score)
            self.tree.current_path.append(current.id)
        
        return self._get_best_thought()
    
    async def _explore_beam(
        self,
        context: AgentContext,
        observation: PageObservation,
    ) -> Thought:
        """Beam search - keep top-k paths at each depth."""
        beam_width = self.config.tot_beam_width
        current_beam = [self.tree.thoughts[rid] for rid in self.tree.root_ids]
        
        for depth in range(self.config.tot_max_depth):
            if not current_beam:
                break
            
            # Explore all thoughts in current beam
            next_candidates = []
            
            for thought in current_beam:
                if thought.status == ThoughtStatus.PENDING:
                    await self._explore_thought(thought, context, observation)
                
                if thought.status == ThoughtStatus.SUCCEEDED:
                    return thought
                
                # Generate children
                children = await self.generator.generate(
                    context, observation, parent_thought=thought
                )
                children = await self.evaluator.evaluate_batch(children, context, observation)
                
                for child in children:
                    self.tree.add_thought(child)
                    next_candidates.append(child)
            
            # Keep top-k for next iteration
            next_candidates.sort(key=lambda t: t.score, reverse=True)
            current_beam = next_candidates[:beam_width]
        
        return self._get_best_thought()
    
    async def _explore_thought(
        self,
        thought: Thought,
        context: AgentContext,
        observation: PageObservation,
    ) -> Optional[Any]:
        """
        Explore a single thought by executing it.
        
        Returns the result of execution, if any.
        """
        self._exploration_count += 1
        thought.status = ThoughtStatus.EXPLORING
        
        logger.debug(f"Exploring: {thought.content[:60]}... (score: {thought.score:.2f})")
        
        if not self.action_executor:
            # Without executor, just mark as pending for external execution
            thought.status = ThoughtStatus.PENDING
            return None
        
        try:
            result = await self.action_executor(thought, context, observation)
            
            if result.get("success"):
                thought.status = ThoughtStatus.SUCCEEDED
                thought.result = str(result.get("data", "Success"))
                self._best_result = result.get("data")
            else:
                thought.status = ThoughtStatus.FAILED
                thought.error = result.get("error", "Unknown error")
            
            thought.observations.append(f"Executed at depth {thought.depth}")
            
            # Record for learning
            self.evaluator.record_outcome(thought)
            
            return result
            
        except Exception as e:
            logger.error(f"Thought execution failed: {e}")
            thought.status = ThoughtStatus.FAILED
            thought.error = str(e)
            return None
    
    def _get_best_thought(self) -> Thought:
        """Get the best thought found so far."""
        # Prefer succeeded thoughts
        succeeded = [t for t in self.tree.thoughts.values() if t.status == ThoughtStatus.SUCCEEDED]
        if succeeded:
            return max(succeeded, key=lambda t: t.score)
        
        # Otherwise return highest-scoring overall
        return max(self.tree.thoughts.values(), key=lambda t: t.score)
    
    async def reflect(
        self,
        context: AgentContext,
        observation: PageObservation,
    ) -> str:
        """
        Self-reflection after ToT exploration.
        
        Analyzes what worked, what didn't, and suggests improvements
        for future explorations.
        """
        if not self.config.tot_self_reflection:
            return ""
        
        # Build summary of exploration
        succeeded = [t for t in self.tree.thoughts.values() if t.status == ThoughtStatus.SUCCEEDED]
        failed = [t for t in self.tree.thoughts.values() if t.status == ThoughtStatus.FAILED]
        
        reflection_prompt = f"""Reflect on the following exploration attempt:

GOAL: {context.goal.objective}

SUCCESSFUL APPROACHES:
{chr(10).join(f'- {t.content}' for t in succeeded) or 'None'}

FAILED APPROACHES:
{chr(10).join(f'- {t.content}: {t.error or "Unknown"}' for t in failed[:5]) or 'None'}

What patterns do you notice? What should be tried differently next time?
Provide 2-3 key insights for improving future explorations.
"""
        
        try:
            response = await self.llm(reflection_prompt)
            logger.info(f"Reflection: {response[:200]}")
            return response
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return ""
