"""
AWE Core State Management
=========================
Manages the shared state machine for web exploration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import asyncio
import logging
from contextlib import asynccontextmanager

from .types import (
    AgentContext,
    AgentMessage,
    AgentRole,
    ExplorationGoal,
    ExplorationPhase,
    ExplorationResult,
    ExtractionResult,
    LearnedPattern,
    PageObservation,
    ThoughtTree,
)


logger = logging.getLogger(__name__)


# =============================================================================
# State Machine
# =============================================================================

class ExplorationStateMachine:
    """
    State machine for web exploration.
    
    State Flow:
    OBSERVE -> THINK -> PLAN -> ACT -> EXTRACT -> VALIDATE -> LEARN
                 ^                                    |
                 |____________________________________| (if more items)
    """
    
    TRANSITIONS = {
        ExplorationPhase.OBSERVE: [ExplorationPhase.THINK, ExplorationPhase.COMPLETE],
        ExplorationPhase.THINK: [ExplorationPhase.PLAN, ExplorationPhase.OBSERVE],
        ExplorationPhase.PLAN: [ExplorationPhase.ACT, ExplorationPhase.THINK],
        ExplorationPhase.ACT: [ExplorationPhase.EXTRACT, ExplorationPhase.OBSERVE, ExplorationPhase.PLAN],
        ExplorationPhase.EXTRACT: [ExplorationPhase.VALIDATE, ExplorationPhase.ACT],
        ExplorationPhase.VALIDATE: [ExplorationPhase.LEARN, ExplorationPhase.EXTRACT, ExplorationPhase.OBSERVE],
        ExplorationPhase.LEARN: [ExplorationPhase.OBSERVE, ExplorationPhase.COMPLETE],
        ExplorationPhase.COMPLETE: [],
    }
    
    def __init__(self, initial_phase: ExplorationPhase = ExplorationPhase.OBSERVE):
        self.current_phase = initial_phase
        self.history: List[ExplorationPhase] = [initial_phase]
        self._listeners: List[Callable[[ExplorationPhase, ExplorationPhase], None]] = []
    
    def can_transition(self, to_phase: ExplorationPhase) -> bool:
        """Check if transition is valid."""
        return to_phase in self.TRANSITIONS.get(self.current_phase, [])
    
    def transition(self, to_phase: ExplorationPhase) -> bool:
        """
        Transition to a new phase.
        Returns True if successful, False if invalid transition.
        """
        if not self.can_transition(to_phase):
            logger.warning(
                f"Invalid transition: {self.current_phase} -> {to_phase}. "
                f"Valid: {self.TRANSITIONS.get(self.current_phase, [])}"
            )
            return False
        
        old_phase = self.current_phase
        self.current_phase = to_phase
        self.history.append(to_phase)
        
        # Notify listeners
        for listener in self._listeners:
            listener(old_phase, to_phase)
        
        logger.debug(f"Transitioned: {old_phase} -> {to_phase}")
        return True
    
    def on_transition(self, callback: Callable[[ExplorationPhase, ExplorationPhase], None]):
        """Register a callback for state transitions."""
        self._listeners.append(callback)
    
    def get_valid_transitions(self) -> List[ExplorationPhase]:
        """Get valid transitions from current state."""
        return self.TRANSITIONS.get(self.current_phase, [])
    
    def is_complete(self) -> bool:
        """Check if exploration is complete."""
        return self.current_phase == ExplorationPhase.COMPLETE


# =============================================================================
# Shared State Store
# =============================================================================

class StateStore:
    """
    Thread-safe shared state store for all agents.
    """
    
    def __init__(self, goal: ExplorationGoal):
        self.goal = goal
        self._lock = asyncio.Lock()
        
        # Core state
        self._context = AgentContext(
            goal=goal,
            current_phase=ExplorationPhase.OBSERVE,
            start_time=datetime.now(),
        )
        
        # Message queue for agent communication
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        
        # State machine
        self._state_machine = ExplorationStateMachine()
        
        # Observation cache
        self._observations: Dict[str, PageObservation] = {}
        
        # Thought tree for ToT
        self._thought_tree = ThoughtTree()
        
        # Extraction results
        self._results: List[ExtractionResult] = []
        
        # Learned patterns
        self._patterns: List[LearnedPattern] = []
        
        # URL tracking
        self._urls_discovered: Set[str] = set()
        self._urls_to_visit: List[str] = []
        self._urls_visited: Set[str] = set()
        self._urls_failed: Set[str] = set()
        
        # Error tracking
        self._errors: List[Dict[str, Any]] = []
        self._retries: Dict[str, int] = {}  # url -> retry count
        
        # Metrics
        self._metrics: Dict[str, Any] = {
            "pages_processed": 0,
            "items_extracted": 0,
            "thoughts_explored": 0,
            "patterns_learned": 0,
            "errors": 0,
        }
    
    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------
    
    async def get_context(self) -> AgentContext:
        """Get current agent context (thread-safe copy)."""
        async with self._lock:
            return AgentContext(
                goal=self.goal,
                current_phase=self._state_machine.current_phase,
                observation=self._observations.get(self._get_current_url()),
                thought_tree=self._thought_tree,
                extracted_items=list(self._results),
                learned_patterns=list(self._patterns),
                urls_discovered=set(self._urls_discovered),
                urls_visited=set(self._urls_visited),
                urls_failed=set(self._urls_failed),
                errors=list(e.get("message", "") for e in self._errors),
                retries=sum(self._retries.values()),
                start_time=self._context.start_time,
                pages_processed=self._metrics["pages_processed"],
            )
    
    def _get_current_url(self) -> str:
        """Get the current URL being processed."""
        if self._urls_to_visit:
            return self._urls_to_visit[0]
        return self.goal.start_url
    
    # -------------------------------------------------------------------------
    # State Machine
    # -------------------------------------------------------------------------
    
    async def transition(self, phase: ExplorationPhase) -> bool:
        """Transition to a new phase."""
        async with self._lock:
            return self._state_machine.transition(phase)
    
    async def get_phase(self) -> ExplorationPhase:
        """Get current phase."""
        async with self._lock:
            return self._state_machine.current_phase
    
    async def is_complete(self) -> bool:
        """Check if exploration is complete."""
        async with self._lock:
            return self._state_machine.is_complete()
    
    # -------------------------------------------------------------------------
    # URL Management
    # -------------------------------------------------------------------------
    
    async def add_discovered_urls(self, urls: List[str]) -> int:
        """Add discovered URLs. Returns count of new URLs added."""
        async with self._lock:
            new_count = 0
            for url in urls:
                if url not in self._urls_discovered:
                    self._urls_discovered.add(url)
                    if url not in self._urls_visited and url not in self._urls_failed:
                        self._urls_to_visit.append(url)
                        new_count += 1
            return new_count
    
    async def get_next_url(self) -> Optional[str]:
        """Get next URL to visit."""
        async with self._lock:
            while self._urls_to_visit:
                url = self._urls_to_visit.pop(0)
                if url not in self._urls_visited:
                    return url
            return None
    
    async def mark_url_visited(self, url: str) -> None:
        """Mark URL as visited."""
        async with self._lock:
            self._urls_visited.add(url)
            self._metrics["pages_processed"] += 1
    
    async def mark_url_failed(self, url: str, error: str) -> None:
        """Mark URL as failed."""
        async with self._lock:
            self._urls_failed.add(url)
            self._retries[url] = self._retries.get(url, 0) + 1
            self._errors.append({
                "url": url,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            })
            self._metrics["errors"] += 1
    
    async def get_urls_remaining(self) -> int:
        """Get count of URLs remaining."""
        async with self._lock:
            return len(self._urls_to_visit)
    
    async def should_retry(self, url: str, max_retries: int = 3) -> bool:
        """Check if URL should be retried."""
        async with self._lock:
            return self._retries.get(url, 0) < max_retries
    
    # -------------------------------------------------------------------------
    # Observation Management
    # -------------------------------------------------------------------------
    
    async def save_observation(self, observation: PageObservation) -> None:
        """Save a page observation."""
        async with self._lock:
            self._observations[observation.url] = observation
    
    async def get_observation(self, url: str) -> Optional[PageObservation]:
        """Get observation for a URL."""
        async with self._lock:
            return self._observations.get(url)
    
    async def get_latest_observation(self) -> Optional[PageObservation]:
        """Get the most recent observation."""
        async with self._lock:
            if not self._observations:
                return None
            # Return observation for current URL or most recent
            current_url = self._get_current_url()
            if current_url in self._observations:
                return self._observations[current_url]
            # Return most recent by timestamp
            return max(self._observations.values(), key=lambda o: o.timestamp)
    
    # -------------------------------------------------------------------------
    # Thought Tree Management (ToT)
    # -------------------------------------------------------------------------
    
    async def get_thought_tree(self) -> ThoughtTree:
        """Get the thought tree."""
        async with self._lock:
            return self._thought_tree
    
    async def update_thought_tree(self, tree: ThoughtTree) -> None:
        """Update the thought tree."""
        async with self._lock:
            self._thought_tree = tree
            self._metrics["thoughts_explored"] = len(tree.thoughts)
    
    # -------------------------------------------------------------------------
    # Results Management
    # -------------------------------------------------------------------------
    
    async def add_result(self, result: ExtractionResult) -> None:
        """Add an extraction result."""
        async with self._lock:
            self._results.append(result)
            if result.is_valid:
                self._metrics["items_extracted"] += 1
    
    async def get_results(self) -> List[ExtractionResult]:
        """Get all extraction results."""
        async with self._lock:
            return list(self._results)
    
    async def get_result_count(self) -> int:
        """Get count of successful extractions."""
        async with self._lock:
            return self._metrics["items_extracted"]
    
    # -------------------------------------------------------------------------
    # Pattern Learning
    # -------------------------------------------------------------------------
    
    async def add_pattern(self, pattern: LearnedPattern) -> None:
        """Add a learned pattern."""
        async with self._lock:
            # Update if exists, otherwise add
            for i, p in enumerate(self._patterns):
                if p.site_domain == pattern.site_domain and p.page_type == pattern.page_type:
                    self._patterns[i] = pattern
                    return
            self._patterns.append(pattern)
            self._metrics["patterns_learned"] += 1
    
    async def get_patterns_for_domain(self, domain: str) -> List[LearnedPattern]:
        """Get learned patterns for a domain."""
        async with self._lock:
            return [p for p in self._patterns if p.site_domain == domain]
    
    # -------------------------------------------------------------------------
    # Message Queue (Agent Communication)
    # -------------------------------------------------------------------------
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to the queue."""
        await self._message_queue.put(message)
    
    async def receive_message(self, timeout: float = None) -> Optional[AgentMessage]:
        """Receive a message from the queue."""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=timeout
                )
            return self._message_queue.get_nowait()
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None
    
    # -------------------------------------------------------------------------
    # Finalization
    # -------------------------------------------------------------------------
    
    async def get_final_result(self) -> ExplorationResult:
        """Build the final exploration result."""
        async with self._lock:
            elapsed = (datetime.now() - self._context.start_time).total_seconds() if self._context.start_time else 0
            
            return ExplorationResult(
                goal=self.goal,
                success=self._metrics["items_extracted"] > 0,
                items=[r.data for r in self._results if r.is_valid],
                patterns=list(self._patterns),
                pages_visited=len(self._urls_visited),
                items_extracted=self._metrics["items_extracted"],
                errors_encountered=self._metrics["errors"],
                duration_seconds=elapsed,
                log=[],  # Could populate with logging
            )


# =============================================================================
# State Context Manager
# =============================================================================

@asynccontextmanager
async def exploration_context(goal: ExplorationGoal):
    """
    Context manager for an exploration session.
    
    Usage:
        async with exploration_context(goal) as state:
            # Use state store
            await state.add_discovered_urls([...])
    """
    state = StateStore(goal)
    try:
        yield state
    finally:
        # Cleanup if needed
        pass
