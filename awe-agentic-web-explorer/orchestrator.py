"""
AWE Orchestrator
================
Main coordinator that ties all agents together for web exploration.

The Orchestrator:
1. Manages the exploration lifecycle
2. Coordinates agent communication
3. Handles state transitions
4. Provides retry/recovery logic
5. Reports progress
"""

from __future__ import annotations
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Browser, Page

from .core.types import (
    AgentContext,
    AgentRole,
    ExplorationGoal,
    ExplorationPhase,
    ExplorationResult,
    PageType,
)
from .core.state import StateStore, ExplorationStateMachine
from .core.config import AWEConfig, DEFAULT_CONFIG
from .agents import (
    AgentPool,
    ObserverAgent,
    PlannerAgent,
    ExecutorAgent,
    ExtractorAgent,
    ValidatorAgent,
    LearnerAgent,
)
from .reasoning import ToTEngine


logger = logging.getLogger(__name__)


class WebExplorer:
    """
    Main entry point for AWE web exploration.
    
    Coordinates all agents to achieve exploration goals using
    Tree of Thought reasoning for intelligent decision making.
    
    Usage:
        async with WebExplorer(model="gemma3:12b") as explorer:
            result = await explorer.explore(goal)
    """
    
    def __init__(
        self,
        model: str = "gemma3:12b",
        config: Optional[AWEConfig] = None,
        llm_func: Optional[Callable] = None,
        vision_func: Optional[Callable] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the WebExplorer.
        
        Args:
            model: LLM model name (for Ollama)
            config: AWE configuration (uses defaults if None)
            llm_func: Custom LLM function (auto-creates one if None)
            vision_func: Custom vision LLM function
            verbose: Whether to print progress
            **kwargs: Additional config overrides
        """
        self.config = config or AWEConfig(**kwargs)
        self.config.model = model
        self.verbose = verbose
        
        # LLM functions
        self.llm_func = llm_func or self._create_llm_func(model)
        self.vision_func = vision_func or self.llm_func
        
        # Browser
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        
        # Agents
        self._pool = AgentPool()
        self._state: Optional[StateStore] = None
        
        # Progress tracking
        self._progress_callback: Optional[Callable] = None
    
    def _create_llm_func(self, model: str) -> Callable:
        """Create an LLM function for the specified model."""
        async def ollama_llm(prompt: str, image_base64: Optional[str] = None, **kwargs) -> str:
            """Call Ollama LLM."""
            import httpx
            
            messages = [{"role": "user", "content": prompt}]
            
            # Add image if provided (for vision models)
            if image_base64:
                messages = [{
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }]
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.config.ollama_base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
        
        return ollama_llm
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
    
    async def _initialize(self):
        """Initialize browser and agents."""
        self.log("Initializing WebExplorer...")
        
        # Start browser
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
            ]
        )
        
        self.log("Browser started")
    
    async def _cleanup(self):
        """Cleanup resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        self.log("Cleanup complete")
    
    def _setup_agents(self, state: StateStore):
        """Setup all agents with shared state."""
        self._pool = AgentPool()
        
        # Observer - vision + DOM analysis
        observer = ObserverAgent(
            config=self.config,
            llm_func=self.llm_func,
            vision_func=self.vision_func,
            state=state,
        )
        self._pool.register(observer)
        
        # Planner - ToT-based strategy
        planner = PlannerAgent(
            config=self.config,
            llm_func=self.llm_func,
            state=state,
        )
        self._pool.register(planner)
        
        # Executor - Playwright actions
        executor = ExecutorAgent(
            config=self.config,
            browser=self._browser,
            state=state,
        )
        self._pool.register(executor)
        
        # Extractor - data extraction
        extractor = ExtractorAgent(
            config=self.config,
            llm_func=self.llm_func,
            state=state,
        )
        self._pool.register(extractor)
        
        # Validator - quality checks
        validator = ValidatorAgent(
            config=self.config,
            llm_func=self.llm_func,
            state=state,
        )
        self._pool.register(validator)
        
        # Learner - pattern learning
        learner = LearnerAgent(
            config=self.config,
            llm_func=self.llm_func,
            state=state,
        )
        self._pool.register(learner)
    
    async def explore(
        self,
        goal: ExplorationGoal,
        progress_callback: Optional[Callable] = None,
    ) -> ExplorationResult:
        """
        Main exploration method.
        
        Coordinates all agents to achieve the exploration goal.
        
        Args:
            goal: ExplorationGoal defining what to extract
            progress_callback: Optional callback for progress updates
        
        Returns:
            ExplorationResult with extracted data and learned patterns
        """
        self._progress_callback = progress_callback
        
        self.log("=" * 60)
        self.log("🌐 AWE WEB EXPLORER")
        self.log("=" * 60)
        self.log(f"Goal: {goal.objective}")
        self.log(f"URL: {goal.start_url}")
        self.log(f"Fields: {', '.join(goal.target_fields)}")
        self.log("")
        
        # Create state store
        self._state = StateStore(goal)
        self._setup_agents(self._state)
        
        # Create page
        self._page = await self._browser.new_page(
            viewport={'width': self.config.viewport[0], 'height': self.config.viewport[1]},
            user_agent=self.config.user_agent,
        )
        
        # Get agents
        observer: ObserverAgent = self._pool.get(AgentRole.OBSERVER)
        planner: PlannerAgent = self._pool.get(AgentRole.PLANNER)
        executor: ExecutorAgent = self._pool.get(AgentRole.EXECUTOR)
        extractor: ExtractorAgent = self._pool.get(AgentRole.EXTRACTOR)
        validator: ValidatorAgent = self._pool.get(AgentRole.VALIDATOR)
        learner: LearnerAgent = self._pool.get(AgentRole.LEARNER)
        
        executor.set_page(self._page)
        
        # Load existing patterns
        if self.config.learning_enabled:
            learner.load_patterns()
        
        try:
            # =========================================================
            # PHASE 1: Initial Observation
            # =========================================================
            self.log("📊 PHASE 1: Observing Start Page")
            self.log("-" * 40)
            
            await self._page.goto(goal.start_url, wait_until='networkidle', timeout=self.config.timeout)
            await self._page.wait_for_timeout(self.config.wait_after_navigation)
            
            observation = await observer.observe(self._page, goal.start_url)
            await self._state.save_observation(observation)
            
            self.log(f"Page type: {observation.page_type.value}")
            self.log(f"Content loading: {observation.content_loading.value}")
            self.log(f"Profile links found: {observation.profile_links}")
            self.log(f"AJAX endpoints: {len(observation.ajax_endpoints)}")
            
            # =========================================================
            # PHASE 2: Planning (with ToT)
            # =========================================================
            self.log("")
            self.log("🧠 PHASE 2: Planning Strategy (ToT)")
            self.log("-" * 40)
            
            context = await self._state.get_context()
            context.observation = observation
            
            strategy = await planner.plan(context, observation)
            
            self.log(f"Selected strategy: {strategy.name}")
            self.log(f"Confidence: {strategy.confidence:.0%}")
            self.log(f"Actions: {len(strategy.actions)}")
            
            # =========================================================
            # PHASE 3: Discovery (Execute Strategy to Find URLs)
            # =========================================================
            self.log("")
            self.log("🔍 PHASE 3: Discovering Items")
            self.log("-" * 40)
            
            discovered_urls = await self._execute_discovery(
                executor, strategy, observation
            )
            
            await self._state.add_discovered_urls(discovered_urls)
            
            self.log(f"Discovered {len(discovered_urls)} item URLs")
            
            # Apply limits
            max_items = goal.constraints.get("max_items", self.config.max_items)
            if len(discovered_urls) > max_items:
                discovered_urls = discovered_urls[:max_items]
                self.log(f"Limited to {max_items} items")
            
            # =========================================================
            # PHASE 4: Extraction
            # =========================================================
            self.log("")
            self.log("📦 PHASE 4: Extracting Data")
            self.log("-" * 40)
            
            # Get domain for pattern lookup
            domain = urlparse(goal.start_url).netloc
            pattern = learner.get_pattern(domain) if self.config.learning_enabled else None
            
            for i, url in enumerate(discovered_urls):
                self.log(f"[{i+1}/{len(discovered_urls)}] {url.split('/')[-1]}")
                
                try:
                    await self._page.goto(url, wait_until='networkidle', timeout=self.config.timeout)
                    await self._page.wait_for_timeout(500)
                    
                    result = await extractor.extract(
                        self._page,
                        url,
                        fields=goal.target_fields,
                        pattern=pattern,
                    )
                    
                    if result.is_valid:
                        self.log(f"  ✓ {result.data.get('name', 'Unknown')}")
                    else:
                        self.log(f"  ⚠ Invalid: {result.validation_errors}")
                    
                    # Rate limiting
                    await asyncio.sleep(self.config.request_delay)
                    
                except Exception as e:
                    self.log(f"  ✗ Error: {e}")
                    await self._state.mark_url_failed(url, str(e))
                
                # Progress callback
                if self._progress_callback:
                    self._progress_callback({
                        "phase": "extraction",
                        "current": i + 1,
                        "total": len(discovered_urls),
                    })
            
            # =========================================================
            # PHASE 5: Validation
            # =========================================================
            self.log("")
            self.log("✅ PHASE 5: Validating Results")
            self.log("-" * 40)
            
            results = await self._state.get_results()
            validation = validator.validate_batch(results)
            
            self.log(f"Passed: {validation['passed']}/{validation['total']}")
            self.log(f"Quality: {validation['avg_quality']:.0%}")
            
            # =========================================================
            # PHASE 6: Learning
            # =========================================================
            if self.config.learning_enabled:
                self.log("")
                self.log("📚 PHASE 6: Learning Patterns")
                self.log("-" * 40)
                
                valid_results = [r for r in results if r.is_valid]
                patterns = await learner.learn_from_results(valid_results)
                
                self.log(f"Learned {len(patterns)} patterns")
                
                if patterns and self.config.save_templates:
                    for p in patterns:
                        path = learner.export_template_file(p)
                        self.log(f"Saved template: {path}")
                
                learner.save_patterns()
            
            # =========================================================
            # Final Result
            # =========================================================
            self.log("")
            self.log("=" * 60)
            self.log("✨ EXPLORATION COMPLETE")
            self.log("=" * 60)
            
            final_result = await self._state.get_final_result()
            final_result.patterns = list(learner.get_all_patterns().values())
            
            self.log(f"Items extracted: {final_result.items_extracted}")
            self.log(f"Pages visited: {final_result.pages_visited}")
            self.log(f"Duration: {final_result.duration_seconds:.1f}s")
            self.log(f"Patterns learned: {len(final_result.patterns)}")
            
            return final_result
            
        finally:
            if self._page:
                await self._page.close()
    
    async def _execute_discovery(
        self,
        executor: ExecutorAgent,
        strategy,
        observation,
    ) -> List[str]:
        """Execute discovery strategy to find item URLs."""
        discovered = []
        
        # First, try to get URLs directly from observation elements
        if observation.elements:
            for elem in observation.elements:
                if elem.looks_like_link and elem.href:
                    # Filter out non-profile URLs
                    href_lower = elem.href.lower()
                    # Skip navigation links, anchors, and program pages
                    skip_patterns = [
                        '#', 'javascript:', 
                        'undergraduate', 'graduate', 
                        'program', 'about', 'contact',
                        'news', 'events', 'research-',
                        'index.', 'home',
                    ]
                    if any(skip in href_lower for skip in skip_patterns):
                        continue
                    discovered.append(elem.href)
        
        # If we found URLs from observation, use those
        if discovered:
            # Deduplicate
            seen = set()
            unique = []
            for url in discovered:
                if url not in seen:
                    seen.add(url)
                    unique.append(url)
            return unique
        
        # Otherwise, execute the strategy
        if strategy.name == "ajax_all":
            # Execute AJAX strategy
            for action in strategy.actions:
                result = await executor.execute_action(action)
                
                if result.success and result.data:
                    if result.data.get("type") == "json":
                        discovered.extend(result.data.get("urls", []))
        
        elif strategy.name == "pagination_crawl":
            # Execute pagination strategy
            page_num = 1
            max_pages = observation.pagination_info.get("total_pages", 10)
            
            while page_num <= max_pages:
                result = await executor.execute_action(
                    strategy.actions[0]  # Extract action
                )
                
                if result.success and result.data:
                    links = result.data.get("links", [])
                    new_urls = [l["url"] for l in links if l.get("url")]
                    
                    if not new_urls or all(u in discovered for u in new_urls):
                        break
                    
                    discovered.extend(new_urls)
                
                # Try to go to next page
                if page_num < max_pages and len(strategy.actions) > 1:
                    next_result = await executor.execute_action(strategy.actions[1])
                    if not next_result.success:
                        break
                
                page_num += 1
        
        elif strategy.name == "scroll_load":
            # Scroll and extract
            scroll_result = await executor.scroll_to_bottom()
            
            result = await executor.execute_action(strategy.actions[0])
            if result.success and result.data:
                links = result.data.get("links", [])
                discovered = [l["url"] for l in links if l.get("url")]
        
        else:
            # Direct extraction
            result = await executor.execute_action(strategy.actions[0])
            if result.success and result.data:
                links = result.data.get("links", [])
                discovered = [l["url"] for l in links if l.get("url")]
        
        # Deduplicate
        seen = set()
        unique = []
        for url in discovered:
            if url not in seen:
                seen.add(url)
                unique.append(url)
        
        return unique
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        if self.verbose:
            print(message)
        
        log_func = getattr(logger, level, logger.info)
        log_func(message)
    
    def on_progress(self, callback: Callable):
        """Set progress callback."""
        self._progress_callback = callback


# =============================================================================
# Convenience Functions
# =============================================================================

async def quick_explore(
    url: str,
    fields: List[str],
    model: str = "gemma3:12b",
    **kwargs,
) -> ExplorationResult:
    """
    Quick one-shot exploration.
    
    Usage:
        result = await quick_explore(
            url="https://example.edu/faculty/",
            fields=["name", "email", "title"],
        )
    """
    goal = ExplorationGoal(
        objective=f"Extract data from {url}",
        target_fields=fields,
        start_url=url,
        constraints=kwargs.get("constraints", {}),
    )
    
    async with WebExplorer(model=model, **kwargs) as explorer:
        return await explorer.explore(goal)
