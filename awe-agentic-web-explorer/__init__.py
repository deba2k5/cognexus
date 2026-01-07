"""
AWE - Agentic Web Explorer
===========================

A framework that enables Small Language Models (SLMs) to perform
complex web exploration tasks through multi-agent collaboration
and Tree of Thought reasoning.

Usage:
    from awe import WebExplorer, ExplorationGoal, quick_explore

    # Option 1: Full control
    goal = ExplorationGoal(
        objective="Extract faculty profiles",
        target_fields=["name", "email", "title", "research_interests"],
        start_url="https://example.edu/faculty/",
    )
    
    async with WebExplorer(model="gemma3:12b") as explorer:
        result = await explorer.explore(goal)
    
    # Option 2: Quick exploration
    result = await quick_explore(
        url="https://example.edu/faculty/",
        fields=["name", "email", "title"],
    )

Architecture:
    - Core: Types, State, Config
    - Reasoning: Tree of Thought engine
    - Agents: Observer, Planner, Executor, Extractor, Validator, Learner
    - Orchestrator: Coordinates all agents
"""

__version__ = "0.1.0"
__author__ = "AWE Framework"

# Core types and models
from .core.types import (
    # Enums
    AgentRole,
    ExplorationPhase,
    PageType,
    ContentLoadingType,
    ActionType,
    ThoughtStatus,
    
    # Data classes
    ExplorationGoal,
    Thought,
    ThoughtTree,
    PageObservation,
    Action,
    ExtractionResult,
    LearnedPattern,
    ExplorationResult,
    AgentContext,
    DOMElement,
    AgentMessage,
    PlaywrightTemplate,
    
    # Pydantic models (for LLM parsing)
    PageAnalysisOutput,
    ExtractionOutput,
    ThoughtGenerationOutput,
)

# State management
from .core.state import StateStore, ExplorationStateMachine

# Configuration
from .core.config import AWEConfig, DEFAULT_CONFIG, PRESETS

# Reasoning engine
from .reasoning.tot import ToTEngine, ThoughtGenerator, ThoughtEvaluator

# Agents
from .agents import (
    BaseAgent,
    AgentPool,
    ObserverAgent,
    PlannerAgent,
    ExecutorAgent,
    ExtractorAgent,
    ValidatorAgent,
    LearnerAgent,
)

# Knowledge graph
from .knowledge import KnowledgeGraph, DomainKnowledge, SiteNode

# Main orchestrator
from .orchestrator import WebExplorer, quick_explore


__all__ = [
    # Version
    "__version__",
    
    # Main entry points
    "WebExplorer",
    "quick_explore",
    
    # Goal definition
    "ExplorationGoal",
    "ExplorationResult",
    
    # Configuration
    "AWEConfig",
    "DEFAULT_CONFIG",
    "PRESETS",
    
    # State
    "StateStore",
    "ExplorationStateMachine",
    
    # Reasoning
    "ToTEngine",
    
    # Agents
    "ObserverAgent",
    "PlannerAgent",
    "ExecutorAgent",
    "ExtractorAgent",
    "ValidatorAgent",
    "LearnerAgent",
    "AgentPool",
    
    # Types
    "AgentRole",
    "ExplorationPhase",
    "PageType",
    "ContentLoadingType",
    "ActionType",
    "ThoughtStatus",
    "Thought",
    "ThoughtTree",
    "PageObservation",
    "Action",
    "ExtractionResult",
    "LearnedPattern",
    "AgentContext",
    "DOMElement",
    "AgentMessage",
    "PlaywrightTemplate",
    
    # LLM output models
    "PageAnalysisOutput",
    "ExtractionOutput",
    "ThoughtGenerationOutput",
    
    # Knowledge graph
    "KnowledgeGraph",
    "DomainKnowledge",
    "SiteNode",
]
