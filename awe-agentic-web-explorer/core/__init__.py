"""
AWE Core Module
===============
Core types, state management, and configuration for the AWE framework.
"""

from .types import (
    # Enums
    AgentRole,
    ExplorationPhase,
    PageType,
    ContentLoadingType,
    ActionType,
    ThoughtStatus,
    
    # Data models
    ExplorationGoal,
    DOMElement,
    PageObservation,
    Thought,
    ThoughtTree,
    Action,
    ExtractionResult,
    LearnedPattern,
    AgentMessage,
    AgentContext,
    ExplorationResult,
    PlaywrightTemplate,
    
    # Pydantic models
    PageAnalysisOutput,
    ExtractionOutput,
    ThoughtGenerationOutput,
    ThoughtEvaluationOutput,
)

from .state import (
    ExplorationStateMachine,
    StateStore,
    exploration_context,
)

from .config import (
    AWEConfig,
    DEFAULT_CONFIG,
    PRESETS,
)


__all__ = [
    # Enums
    "AgentRole",
    "ExplorationPhase",
    "PageType",
    "ContentLoadingType",
    "ActionType",
    "ThoughtStatus",
    
    # Data models
    "ExplorationGoal",
    "DOMElement",
    "PageObservation",
    "Thought",
    "ThoughtTree",
    "Action",
    "ExtractionResult",
    "LearnedPattern",
    "AgentMessage",
    "AgentContext",
    "ExplorationResult",
    "PlaywrightTemplate",
    
    # Pydantic models
    "PageAnalysisOutput",
    "ExtractionOutput",
    "ThoughtGenerationOutput",
    "ThoughtEvaluationOutput",
    
    # State management
    "ExplorationStateMachine",
    "StateStore",
    "exploration_context",
    
    # Config
    "AWEConfig",
    "DEFAULT_CONFIG",
    "PRESETS",
]
