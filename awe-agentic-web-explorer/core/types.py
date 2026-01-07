"""
AWE Core Types & Data Models
============================
Central type definitions for the Agentic Web Explorer framework.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
import json


# =============================================================================
# Enums
# =============================================================================

class AgentRole(str, Enum):
    """Types of agents in the AWE framework."""
    OBSERVER = "observer"      # Vision + DOM analysis
    PLANNER = "planner"        # Strategy selection (ToT)
    EXECUTOR = "executor"      # Action execution (Playwright)
    EXTRACTOR = "extractor"    # Data extraction
    VALIDATOR = "validator"    # Quality assurance
    LEARNER = "learner"        # Template/pattern learning


class ExplorationPhase(str, Enum):
    """Phases of web exploration."""
    OBSERVE = "observe"        # Understand the page
    THINK = "think"           # Generate thoughts (ToT)
    PLAN = "plan"             # Select strategy
    ACT = "act"               # Execute actions
    EXTRACT = "extract"       # Pull data
    VALIDATE = "validate"     # Check quality
    LEARN = "learn"           # Save patterns
    COMPLETE = "complete"     # Done


class PageType(str, Enum):
    """Types of web pages encountered."""
    DIRECTORY = "directory"       # List of items (faculty list)
    PROFILE = "profile"           # Individual detail page
    SEARCH = "search"             # Search interface
    FORM = "form"                 # Input form
    LOGIN = "login"               # Authentication page
    NAVIGATION = "navigation"     # Index/sitemap
    CONTENT = "content"           # Article/blog
    UNKNOWN = "unknown"


class ContentLoadingType(str, Enum):
    """How content is loaded on the page."""
    STATIC = "static"             # All content in initial HTML
    AJAX_ON_LOAD = "ajax_on_load" # Fetches data on page load
    AJAX_ON_SCROLL = "ajax_on_scroll"  # Infinite scroll
    AJAX_ON_CLICK = "ajax_on_click"    # Click to load more
    PAGINATION_URL = "pagination_url"   # URL-based pagination


class ActionType(str, Enum):
    """Types of browser actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    SCROLL = "scroll"
    TYPE = "type"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    EXTRACT = "extract"
    HOVER = "hover"
    SELECT = "select"


class ThoughtStatus(str, Enum):
    """Status of a thought in ToT."""
    PENDING = "pending"
    EXPLORING = "exploring"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


# =============================================================================
# Core Data Models
# =============================================================================

@dataclass
class ExplorationGoal:
    """
    Defines what the explorer should achieve.
    This is the high-level objective provided by the user.
    """
    objective: str                                    # What to accomplish
    target_fields: List[str]                          # Fields to extract
    start_url: str                                    # Where to begin
    constraints: Dict[str, Any] = field(default_factory=dict)  # Limits
    success_criteria: Optional[str] = None            # How to know we're done
    examples: List[Dict[str, Any]] = field(default_factory=list)  # Sample outputs
    
    def to_prompt(self) -> str:
        """Convert goal to a prompt-friendly format."""
        lines = [
            f"OBJECTIVE: {self.objective}",
            f"TARGET FIELDS: {', '.join(self.target_fields)}",
            f"START URL: {self.start_url}",
        ]
        if self.constraints:
            lines.append(f"CONSTRAINTS: {json.dumps(self.constraints)}")
        if self.success_criteria:
            lines.append(f"SUCCESS: {self.success_criteria}")
        if self.examples:
            lines.append(f"EXAMPLE OUTPUT: {json.dumps(self.examples[0], indent=2)}")
        return "\n".join(lines)


@dataclass
class DOMElement:
    """Represents an element in the DOM with analysis metadata."""
    tag: str
    selector: str                                     # CSS selector path
    classes: List[str] = field(default_factory=list)
    id: Optional[str] = None
    text: str = ""
    href: Optional[str] = None
    src: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    bounding_box: Optional[Dict[str, float]] = None   # x, y, width, height
    is_visible: bool = True
    children_count: int = 0
    
    # Analysis flags
    looks_like_link: bool = False
    looks_like_button: bool = False
    looks_like_input: bool = False
    looks_like_card: bool = False
    looks_like_pagination: bool = False


@dataclass
class PageObservation:
    """
    Complete observation of a web page.
    Combines vision (screenshot) and structure (DOM).
    """
    url: str
    title: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Visual understanding
    screenshot_base64: Optional[str] = None
    screenshot_path: Optional[str] = None
    
    # DOM understanding
    html: str = ""
    semantic_dom: str = ""                            # Compressed XML representation
    elements: List[DOMElement] = field(default_factory=list)
    
    # Page classification
    page_type: PageType = PageType.UNKNOWN
    content_loading: ContentLoadingType = ContentLoadingType.STATIC
    
    # Discovered patterns
    link_patterns: List[str] = field(default_factory=list)      # URL patterns found
    card_patterns: List[str] = field(default_factory=list)      # Card CSS patterns
    pagination_info: Dict[str, Any] = field(default_factory=dict)
    ajax_endpoints: List[str] = field(default_factory=list)
    
    # Counts
    total_links: int = 0
    profile_links: int = 0
    visible_items: int = 0


@dataclass
class Thought:
    """
    A reasoning step in Tree of Thought.
    Represents a potential approach/action to try.
    """
    id: str
    content: str                                      # What this thought proposes
    parent_id: Optional[str] = None                   # For tree structure
    depth: int = 0
    
    # Evaluation scores (0.0 to 1.0)
    feasibility: float = 0.5                          # Can we do this?
    confidence: float = 0.5                           # Will it work?
    value: float = 0.5                                # Is it worth trying?
    
    # Status tracking
    status: ThoughtStatus = ThoughtStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    
    # Execution info
    actions_taken: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    
    @property
    def score(self) -> float:
        """Combined score for ranking thoughts."""
        return (self.feasibility * 0.3 + self.confidence * 0.4 + self.value * 0.3)


@dataclass
class ThoughtTree:
    """
    Tree structure for ToT reasoning.
    Enables exploration, backtracking, and path selection.
    """
    thoughts: Dict[str, Thought] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)
    current_path: List[str] = field(default_factory=list)
    
    def add_thought(self, thought: Thought) -> None:
        """Add a thought to the tree."""
        self.thoughts[thought.id] = thought
        if thought.parent_id is None:
            self.root_ids.append(thought.id)
    
    def get_children(self, thought_id: str) -> List[Thought]:
        """Get all children of a thought."""
        return [t for t in self.thoughts.values() if t.parent_id == thought_id]
    
    def get_best_path(self) -> List[Thought]:
        """Get the highest-scoring path through the tree."""
        if not self.root_ids:
            return []
        
        best_path = []
        current = max([self.thoughts[rid] for rid in self.root_ids], key=lambda t: t.score)
        best_path.append(current)
        
        while True:
            children = self.get_children(current.id)
            if not children:
                break
            current = max(children, key=lambda t: t.score)
            best_path.append(current)
        
        return best_path
    
    def backtrack(self) -> Optional[Thought]:
        """Backtrack to last unexplored sibling."""
        while self.current_path:
            current_id = self.current_path[-1]
            parent_id = self.thoughts[current_id].parent_id
            
            if parent_id:
                siblings = self.get_children(parent_id)
                unexplored = [s for s in siblings 
                              if s.status == ThoughtStatus.PENDING and s.id != current_id]
                if unexplored:
                    return max(unexplored, key=lambda t: t.score)
            
            self.current_path.pop()
        
        return None


@dataclass
class Action:
    """
    A concrete action to execute on the page.
    """
    type: ActionType
    target: Optional[str] = None                      # CSS selector or URL
    value: Optional[str] = None                       # For type/select actions
    wait_after: int = 500                             # ms to wait after action
    timeout: int = 30000                              # ms timeout
    screenshot_before: bool = False
    screenshot_after: bool = False
    
    # Result tracking
    executed: bool = False
    success: bool = False
    error: Optional[str] = None
    result_data: Any = None


@dataclass
class ExtractionResult:
    """
    Result of extracting data from a page.
    """
    url: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Extracted data
    data: Dict[str, Any] = field(default_factory=dict)
    raw_html: str = ""
    
    # Quality metrics
    fields_found: int = 0
    fields_expected: int = 0
    confidence: float = 0.0
    
    # Template used (if any)
    template_used: Optional[str] = None
    
    # Validation
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class LearnedPattern:
    """
    A pattern learned from successful extractions.
    Can be converted to Playwright code.
    """
    id: str
    site_domain: str
    page_type: PageType
    created_at: datetime = field(default_factory=datetime.now)
    
    # Selectors that worked
    selectors: Dict[str, str] = field(default_factory=dict)       # field -> selector
    backup_selectors: Dict[str, List[str]] = field(default_factory=dict)
    
    # Extraction methods
    extraction_methods: Dict[str, str] = field(default_factory=dict)  # field -> method
    
    # Page structure info
    page_structure: str = ""
    ajax_endpoint: Optional[str] = None
    pagination_pattern: Optional[str] = None
    
    # Success tracking
    times_used: int = 0
    success_rate: float = 0.0
    last_used: Optional[datetime] = None
    
    def to_playwright_code(self) -> str:
        """Generate Playwright extraction code from this pattern."""
        lines = [
            "async def extract(page: Page) -> Dict[str, Any]:",
            "    data = {}",
        ]
        
        for field_name, selector in self.selectors.items():
            method = self.extraction_methods.get(field_name, "text")
            
            if method == "text":
                lines.append(f"    if el := await page.query_selector('{selector}'):")
                lines.append(f"        data['{field_name}'] = await el.inner_text()")
            elif method == "href":
                lines.append(f"    if el := await page.query_selector('{selector}'):")
                lines.append(f"        data['{field_name}'] = await el.get_attribute('href')")
            elif method == "src":
                lines.append(f"    if el := await page.query_selector('{selector}'):")
                lines.append(f"        data['{field_name}'] = await el.get_attribute('src')")
            elif method == "list":
                lines.append(f"    data['{field_name}'] = []")
                lines.append(f"    for el in await page.query_selector_all('{selector}'):")
                lines.append(f"        data['{field_name}'].append(await el.inner_text())")
        
        lines.append("    return data")
        return "\n".join(lines)


# =============================================================================
# Agent Communication
# =============================================================================

@dataclass
class AgentMessage:
    """
    Message passed between agents.
    """
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str                                 # request, response, error
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None              # Link request/response


@dataclass
class AgentContext:
    """
    Context passed to agents for decision making.
    """
    goal: ExplorationGoal
    current_phase: ExplorationPhase
    observation: Optional[PageObservation] = None
    thought_tree: Optional[ThoughtTree] = None
    extracted_items: List[ExtractionResult] = field(default_factory=list)
    learned_patterns: List[LearnedPattern] = field(default_factory=list)
    
    # Exploration state
    urls_discovered: Set[str] = field(default_factory=set)
    urls_visited: Set[str] = field(default_factory=set)
    urls_failed: Set[str] = field(default_factory=set)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    retries: int = 0
    
    # Performance metrics
    start_time: Optional[datetime] = None
    pages_processed: int = 0


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class ExplorationResult:
    """
    Final result of a web exploration.
    """
    goal: ExplorationGoal
    success: bool
    
    # Extracted data
    items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Patterns learned
    patterns: List[LearnedPattern] = field(default_factory=list)
    
    # Statistics
    pages_visited: int = 0
    items_extracted: int = 0
    errors_encountered: int = 0
    duration_seconds: float = 0.0
    
    # Errors list
    errors: List[str] = field(default_factory=list)
    
    # Execution log
    log: List[str] = field(default_factory=list)
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        data = {
            "goal": {
                "objective": self.goal.objective,
                "target_fields": self.goal.target_fields,
                "start_url": self.goal.start_url,
            },
            "success": self.success,
            "statistics": {
                "pages_visited": self.pages_visited,
                "items_extracted": self.items_extracted,
                "errors": self.errors_encountered,
                "duration_seconds": self.duration_seconds,
            },
            "items": self.items,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_playwright_template(self) -> PlaywrightTemplate:
        """Export learned patterns as Playwright code."""
        if not self.patterns:
            return PlaywrightTemplate(code="# No patterns learned")
        
        # Use the most successful pattern
        best_pattern = max(self.patterns, key=lambda p: p.success_rate)
        return PlaywrightTemplate(
            code=best_pattern.to_playwright_code(),
            pattern=best_pattern
        )


@dataclass
class PlaywrightTemplate:
    """
    A reusable Playwright template for extraction.
    """
    code: str
    pattern: Optional[LearnedPattern] = None
    
    def save(self, path: str) -> None:
        """Save template to Python file."""
        header = '''"""
Auto-generated Playwright extraction template.
Generated by AWE Framework.
"""

from playwright.async_api import Page
from typing import Dict, Any

'''
        with open(path, 'w') as f:
            f.write(header)
            f.write(self.code)
            f.write("\n")


# =============================================================================
# Pydantic Models for LLM Output Parsing
# =============================================================================

class PageAnalysisOutput(BaseModel):
    """LLM output for page analysis."""
    page_type: str = Field(description="One of: directory, profile, search, form, login, navigation, content, unknown")
    content_loading: str = Field(description="One of: static, ajax_on_load, ajax_on_scroll, ajax_on_click, pagination_url")
    key_elements: List[str] = Field(default_factory=list, description="Important CSS selectors found")
    ajax_endpoint: Optional[str] = Field(None, description="AJAX endpoint if detected")
    pagination_pattern: Optional[str] = Field(None, description="How pagination works")
    item_count_estimate: int = Field(0, description="Estimated number of items on page")
    reasoning: str = Field(description="Step-by-step reasoning about the page")


class ExtractionOutput(BaseModel):
    """LLM output for data extraction."""
    data: Dict[str, Any] = Field(default_factory=dict, description="Extracted field values")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in extraction")
    missing_fields: List[str] = Field(default_factory=list, description="Fields not found")
    selector_suggestions: Dict[str, str] = Field(default_factory=dict, description="Suggested selectors for each field")


class ThoughtGenerationOutput(BaseModel):
    """LLM output for ToT thought generation."""
    thoughts: List[Dict[str, Any]] = Field(description="List of possible approaches")
    # Each thought: {content, feasibility, confidence, value, reasoning}


class ThoughtEvaluationOutput(BaseModel):
    """LLM output for evaluating thoughts."""
    thought_id: str = Field(description="ID of thought being evaluated")
    should_explore: bool = Field(description="Whether to explore this thought")
    adjusted_score: float = Field(ge=0.0, le=1.0, description="Updated confidence score")
    reasoning: str = Field(description="Why this evaluation")
