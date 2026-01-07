"""
AWE Core Configuration
======================
Configuration for the Agentic Web Explorer framework.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class AWEConfig:
    """
    Configuration for the AWE framework.
    """
    
    # -------------------------------------------------------------------------
    # LLM Settings
    # -------------------------------------------------------------------------
    
    # Primary model for reasoning
    model: str = "llama-3.3-70b-versatile"
    model_provider: Literal["ollama", "openai", "anthropic", "groq", "litellm"] = "groq"
    
    # Vision model (can be same as model)
    vision_model: str = "llama-3.3-70b-versatile"
    vision_enabled: bool = True
    
    # Model parameters
    temperature: float = 0.3  # Lower for more deterministic output
    max_tokens: int = 4096
    
    # API endpoints (if using remote services)
    ollama_base_url: str = "http://localhost:11434"
    groq_api_key: Optional[str] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # Tree of Thought (ToT) Settings
    # -------------------------------------------------------------------------
    
    tot_enabled: bool = True
    
    # How many alternative thoughts to generate at each step
    tot_max_thoughts: int = 5
    
    # Maximum depth of thought exploration
    tot_max_depth: int = 3
    
    # Search strategy for exploring thought tree
    tot_search_strategy: Literal["bfs", "dfs", "beam"] = "bfs"
    
    # For beam search, how many paths to keep
    tot_beam_width: int = 3
    
    # Minimum score threshold to continue exploring a thought
    tot_min_score: float = 0.3
    
    # Enable self-reflection after failures
    tot_self_reflection: bool = True
    
    # -------------------------------------------------------------------------
    # Browser Settings
    # -------------------------------------------------------------------------
    
    headless: bool = True
    viewport: Tuple[int, int] = (1280, 720)
    timeout: int = 30000  # ms
    wait_after_navigation: int = 1000  # ms
    
    # Browser context options
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    locale: str = "en-US"
    timezone: str = "America/Chicago"
    
    # Screenshot settings
    screenshots_enabled: bool = True
    screenshots_dir: str = "./screenshots"
    screenshot_on_error: bool = True
    
    # -------------------------------------------------------------------------
    # Extraction Settings
    # -------------------------------------------------------------------------
    
    # Whether to use LLM for extraction (vs pure CSS selectors)
    llm_extraction: bool = True
    
    # Fallback to LLM if CSS extraction fails
    llm_fallback: bool = True
    
    # Fields to extract (can be overridden per goal)
    default_fields: List[str] = field(default_factory=lambda: [
        "name", "title", "email", "phone", "department",
        "education", "research_areas", "bio", "image"
    ])
    
    # -------------------------------------------------------------------------
    # Learning Settings
    # -------------------------------------------------------------------------
    
    # Enable pattern learning
    learning_enabled: bool = True
    
    # Path to knowledge graph storage
    knowledge_graph_path: str = "./knowledge"
    
    # Save Playwright templates for reuse
    save_templates: bool = True
    templates_dir: str = "./templates"
    
    # Learn from user corrections
    learn_from_corrections: bool = True
    
    # Minimum success rate to save a pattern
    min_success_rate: float = 0.7
    
    # -------------------------------------------------------------------------
    # Reliability Settings
    # -------------------------------------------------------------------------
    
    max_retries: int = 3
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    
    # Rate limiting
    request_delay: float = 0.5  # seconds between requests
    max_concurrent_pages: int = 5
    
    # Limits
    max_pages: int = 1000
    max_items: int = 10000
    max_duration: int = 3600  # seconds
    
    # -------------------------------------------------------------------------
    # Logging Settings
    # -------------------------------------------------------------------------
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[str] = None
    verbose: bool = True
    
    # Save HTML for debugging
    save_html: bool = True
    html_dir: str = "./html_logs"
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.tot_max_thoughts < 1:
            issues.append("tot_max_thoughts must be >= 1")
        
        if self.tot_max_depth < 1:
            issues.append("tot_max_depth must be >= 1")
        
        if self.viewport[0] < 100 or self.viewport[1] < 100:
            issues.append("viewport dimensions too small")
        
        if self.temperature < 0 or self.temperature > 2:
            issues.append("temperature should be between 0 and 2")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "model_provider": self.model_provider,
            "vision_model": self.vision_model,
            "vision_enabled": self.vision_enabled,
            "tot_enabled": self.tot_enabled,
            "tot_max_thoughts": self.tot_max_thoughts,
            "tot_max_depth": self.tot_max_depth,
            "headless": self.headless,
            "viewport": list(self.viewport),
            "timeout": self.timeout,
            "learning_enabled": self.learning_enabled,
            "max_retries": self.max_retries,
        }


# Default configuration
DEFAULT_CONFIG = AWEConfig()


# Presets for different use cases
PRESETS = {
    "fast": AWEConfig(
        tot_max_thoughts=3,
        tot_max_depth=2,
        timeout=15000,
        llm_extraction=False,
        screenshots_enabled=False,
    ),
    "thorough": AWEConfig(
        tot_max_thoughts=7,
        tot_max_depth=5,
        tot_search_strategy="beam",
        tot_beam_width=5,
        timeout=60000,
        max_retries=5,
    ),
    "development": AWEConfig(
        headless=False,
        verbose=True,
        save_html=True,
        screenshots_enabled=True,
        screenshot_on_error=True,
        log_level="DEBUG",
    ),
}
