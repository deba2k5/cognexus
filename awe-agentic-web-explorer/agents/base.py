"""
AWE Base Agent
==============
Abstract base class for all AWE agents.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.state import StateStore
    from ..core.types import AgentContext, AgentMessage, AgentRole

from ..core.config import AWEConfig


logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Describes a capability an agent has."""
    name: str
    description: str
    required_inputs: List[str]
    outputs: List[str]


class BaseAgent(ABC):
    """
    Abstract base class for all AWE agents.
    
    Each agent:
    - Has a specific role (Observer, Planner, Executor, etc.)
    - Can process messages from other agents
    - Has access to shared state
    - Can use an LLM for decision making
    """
    
    def __init__(
        self,
        role: "AgentRole",
        config: AWEConfig,
        llm_func: Optional[Callable] = None,
        state: Optional["StateStore"] = None,
    ):
        self.role = role
        self.config = config
        self.llm = llm_func
        self.state = state
        
        self._capabilities: List[AgentCapability] = []
        self._is_initialized = False
    
    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        return f"{self.role.value.title()}Agent"
    
    @abstractmethod
    async def process(self, context: "AgentContext") -> Dict[str, Any]:
        """
        Main processing method for the agent.
        
        Args:
            context: Current exploration context
        
        Returns:
            Dict containing the agent's output
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize the agent (called once before first use)."""
        self._is_initialized = True
        logger.debug(f"{self.name} initialized")
    
    async def cleanup(self) -> None:
        """Cleanup resources (called when agent is no longer needed)."""
        self._is_initialized = False
    
    def register_capability(self, capability: AgentCapability) -> None:
        """Register a capability this agent has."""
        self._capabilities.append(capability)
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Get list of agent capabilities."""
        return self._capabilities
    
    async def handle_message(self, message: "AgentMessage") -> Optional["AgentMessage"]:
        """
        Handle an incoming message from another agent.
        
        Default implementation processes the message and returns a response.
        Override for custom message handling.
        """
        logger.debug(f"{self.name} received message from {message.from_agent}")
        
        # Default: process using main process method
        if self.state:
            context = await self.state.get_context()
            result = await self.process(context)
            
            from ..core.types import AgentMessage
            return AgentMessage(
                from_agent=self.role,
                to_agent=message.from_agent,
                message_type="response",
                content=result,
                correlation_id=message.correlation_id,
            )
        
        return None
    
    async def call_llm(self, prompt: str, **kwargs) -> str:
        """
        Call the LLM with the given prompt.
        
        Handles retries and error logging.
        """
        if not self.llm:
            raise ValueError(f"{self.name} has no LLM configured")
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.llm(prompt, **kwargs)
                return response
            except Exception as e:
                logger.warning(f"{self.name} LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_backoff ** attempt)
        
        raise RuntimeError("LLM call failed after all retries")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with the agent's name."""
        log_func = getattr(logger, level, logger.info)
        log_func(f"[{self.name}] {message}")


class AgentPool:
    """
    Manages a pool of agents and coordinates message passing.
    """
    
    def __init__(self):
        self._agents: Dict["AgentRole", BaseAgent] = {}
        self._message_handlers: Dict["AgentRole", List[Callable]] = {}
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the pool."""
        self._agents[agent.role] = agent
        logger.debug(f"Registered {agent.name}")
    
    def get(self, role: "AgentRole") -> Optional[BaseAgent]:
        """Get an agent by role."""
        return self._agents.get(role)
    
    async def initialize_all(self) -> None:
        """Initialize all agents."""
        for agent in self._agents.values():
            await agent.initialize()
    
    async def cleanup_all(self) -> None:
        """Cleanup all agents."""
        for agent in self._agents.values():
            await agent.cleanup()
    
    async def send_message(
        self,
        from_role: "AgentRole",
        to_role: "AgentRole",
        content: Any,
        message_type: str = "request",
    ) -> Optional["AgentMessage"]:
        """
        Send a message from one agent to another.
        
        Returns the response message, if any.
        """
        from ..core.types import AgentMessage
        import uuid
        
        message = AgentMessage(
            from_agent=from_role,
            to_agent=to_role,
            message_type=message_type,
            content=content,
            correlation_id=str(uuid.uuid4()),
        )
        
        target_agent = self._agents.get(to_role)
        if not target_agent:
            logger.warning(f"No agent registered for role {to_role}")
            return None
        
        return await target_agent.handle_message(message)
    
    async def broadcast(
        self,
        from_role: "AgentRole",
        content: Any,
        exclude_roles: Optional[List["AgentRole"]] = None,
    ) -> List["AgentMessage"]:
        """
        Broadcast a message to all agents.
        
        Returns list of responses.
        """
        exclude = set(exclude_roles or [])
        exclude.add(from_role)  # Don't send to self
        
        responses = []
        for role, agent in self._agents.items():
            if role not in exclude:
                response = await self.send_message(from_role, role, content, "broadcast")
                if response:
                    responses.append(response)
        
        return responses
