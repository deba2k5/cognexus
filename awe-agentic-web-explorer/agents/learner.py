"""
AWE Learner Agent
=================
Learns patterns from successful extractions and generates templates.

The Learner:
1. Analyzes successful extractions to find patterns
2. Generates reusable CSS selectors
3. Creates Playwright templates
4. Maintains a pattern library
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.async_api import Page

from .base import BaseAgent, AgentCapability
from ..core.types import (
    AgentContext,
    AgentRole,
    ExtractionResult,
    LearnedPattern,
    PageType,
)
from ..core.config import AWEConfig


logger = logging.getLogger(__name__)


class LearnerAgent(BaseAgent):
    """
    Pattern learning agent.
    
    Responsibilities:
    - Analyze successful extractions
    - Identify reliable selectors
    - Generate reusable patterns
    - Create Playwright templates
    """
    
    def __init__(
        self,
        config: AWEConfig,
        llm_func: Optional[Callable] = None,
        state=None,
    ):
        super().__init__(AgentRole.LEARNER, config, llm_func, state)
        
        # Pattern storage
        self._patterns: Dict[str, LearnedPattern] = {}
        self._selector_stats: Dict[str, Dict[str, int]] = {}  # domain -> selector -> success count
        
        self.register_capability(AgentCapability(
            name="pattern_learning",
            description="Learn extraction patterns from successful results",
            required_inputs=["extraction_results"],
            outputs=["learned_patterns"],
        ))
    
    async def process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Learn patterns from context.
        """
        results = context.extracted_items
        if not results:
            return {"error": "No extraction results to learn from"}
        
        # Filter to valid results
        valid_results = [r for r in results if r.is_valid]
        
        if not valid_results:
            return {"patterns_learned": 0}
        
        # Learn from results
        patterns = await self.learn_from_results(valid_results)
        
        return {
            "patterns_learned": len(patterns),
            "patterns": patterns,
        }
    
    async def learn_from_results(
        self,
        results: List[ExtractionResult],
    ) -> List[LearnedPattern]:
        """
        Learn patterns from a list of extraction results.
        
        Args:
            results: List of successful ExtractionResult objects
        
        Returns:
            List of learned LearnedPattern objects
        """
        if not results:
            return []
        
        # Group by domain
        by_domain: Dict[str, List[ExtractionResult]] = {}
        for result in results:
            domain = urlparse(result.url).netloc
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(result)
        
        patterns = []
        
        for domain, domain_results in by_domain.items():
            if len(domain_results) < 2:
                continue  # Need multiple samples for reliable patterns
            
            pattern = await self._learn_pattern_for_domain(domain, domain_results)
            if pattern:
                patterns.append(pattern)
                self._patterns[domain] = pattern
        
        return patterns
    
    async def _learn_pattern_for_domain(
        self,
        domain: str,
        results: List[ExtractionResult],
    ) -> Optional[LearnedPattern]:
        """Learn a pattern from results for a single domain."""
        self.log(f"Learning pattern for {domain} from {len(results)} samples", "debug")
        
        # Analyze HTML to find consistent selectors
        selectors = {}
        extraction_methods = {}
        backup_selectors = {}
        
        # Common fields to find selectors for
        fields = ["name", "title", "email", "phone", "department", "bio", "image"]
        
        for field in fields:
            # Collect values from results
            values = [r.data.get(field) for r in results if r.data.get(field)]
            
            if not values:
                continue
            
            # Find selectors that work for this field
            # (In practice, we'd analyze the HTML to find what selectors produced these values)
            field_selectors = self._find_selectors_for_field(field, results)
            
            if field_selectors:
                selectors[field] = field_selectors[0]
                if len(field_selectors) > 1:
                    backup_selectors[field] = field_selectors[1:]
                
                # Determine extraction method
                if field == "email":
                    extraction_methods[field] = "href"
                elif field == "phone":
                    extraction_methods[field] = "href"
                elif field == "image":
                    extraction_methods[field] = "src"
                elif field in ["education", "research_areas"]:
                    extraction_methods[field] = "list"
                else:
                    extraction_methods[field] = "text"
        
        if len(selectors) < 2:
            return None  # Not enough patterns found
        
        # Calculate success rate
        valid_count = sum(1 for r in results if r.is_valid)
        success_rate = valid_count / len(results)
        
        pattern = LearnedPattern(
            id=f"{domain}_{datetime.now().strftime('%Y%m%d')}",
            site_domain=domain,
            page_type=PageType.PROFILE,
            selectors=selectors,
            backup_selectors=backup_selectors,
            extraction_methods=extraction_methods,
            times_used=len(results),
            success_rate=success_rate,
            last_used=datetime.now(),
        )
        
        self.log(f"Learned pattern with {len(selectors)} selectors, {success_rate:.0%} success rate", "debug")
        
        return pattern
    
    def _find_selectors_for_field(
        self,
        field: str,
        results: List[ExtractionResult],
    ) -> List[str]:
        """
        Find CSS selectors that work for a field.
        
        This is a simplified implementation - in practice you'd analyze
        the HTML to find what selectors actually produced the values.
        """
        # Common selector patterns per field
        FIELD_SELECTOR_MAP = {
            "name": [
                "h1.masthead__title",
                "h1.faculty-name",
                "h1.profile-name",
                "h1",
            ],
            "title": [
                "h2.masthead__subtitle span",
                "h2.masthead__subtitle",
                ".faculty-title",
                ".position",
                ".title",
            ],
            "email": [
                "a[href^='mailto:']",
            ],
            "phone": [
                "a[href^='tel:']",
            ],
            "department": [
                ".department",
                "[class*='department']",
            ],
            "bio": [
                ".biography p",
                ".bio p",
                ".content > p",
            ],
            "image": [
                "img.profile",
                "img[class*='profile']",
                ".headshot img",
            ],
        }
        
        return FIELD_SELECTOR_MAP.get(field, [])
    
    async def learn_from_page(
        self,
        page: "Page",
        url: str,
        extracted_data: Dict[str, Any],
    ) -> Optional[LearnedPattern]:
        """
        Learn a pattern from a single page extraction.
        
        Analyzes the HTML to find which selectors produced each value.
        """
        domain = urlparse(url).netloc
        html = await page.content()
        
        selectors = {}
        extraction_methods = {}
        backup_selectors = {}
        
        for field, value in extracted_data.items():
            if not value:
                continue
            
            # Find selector for this value
            found_selectors = await self._find_selector_for_value(page, field, value)
            
            if found_selectors:
                selectors[field] = found_selectors[0]
                if len(found_selectors) > 1:
                    backup_selectors[field] = found_selectors[1:]
                
                # Determine method
                if field in ["email", "phone"]:
                    extraction_methods[field] = "href"
                elif field == "image":
                    extraction_methods[field] = "src"
                else:
                    extraction_methods[field] = "text"
        
        if len(selectors) < 2:
            return None
        
        pattern = LearnedPattern(
            id=f"{domain}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            site_domain=domain,
            page_type=PageType.PROFILE,
            selectors=selectors,
            backup_selectors=backup_selectors,
            extraction_methods=extraction_methods,
            times_used=1,
            success_rate=1.0,
            last_used=datetime.now(),
        )
        
        self._patterns[domain] = pattern
        
        return pattern
    
    async def _find_selector_for_value(
        self,
        page: "Page",
        field: str,
        value: Any,
    ) -> List[str]:
        """Find selectors that produce a specific value."""
        selectors = []
        str_value = str(value).strip()[:50]  # Compare first 50 chars
        
        # Try common selectors and see which matches
        candidates = self._find_selectors_for_field(field, [])
        
        for selector in candidates:
            try:
                elem = await page.query_selector(selector)
                if not elem:
                    continue
                
                if field == "email":
                    href = await elem.get_attribute("href")
                    if href and value in href.replace("mailto:", ""):
                        selectors.append(selector)
                elif field == "image":
                    src = await elem.get_attribute("src")
                    if src and value in src:
                        selectors.append(selector)
                else:
                    text = await elem.inner_text()
                    if text and str_value in text.strip():
                        selectors.append(selector)
            except:
                continue
        
        return selectors
    
    def get_pattern(self, domain: str) -> Optional[LearnedPattern]:
        """Get learned pattern for a domain."""
        return self._patterns.get(domain)
    
    def get_all_patterns(self) -> Dict[str, LearnedPattern]:
        """Get all learned patterns."""
        return dict(self._patterns)
    
    def save_patterns(self, path: Optional[str] = None) -> str:
        """
        Save learned patterns to disk.
        
        Returns the path where patterns were saved.
        """
        if path is None:
            path = os.path.join(self.config.knowledge_graph_path, "patterns.json")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {}
        for domain, pattern in self._patterns.items():
            data[domain] = {
                "id": pattern.id,
                "site_domain": pattern.site_domain,
                "page_type": pattern.page_type.value,
                "selectors": pattern.selectors,
                "backup_selectors": pattern.backup_selectors,
                "extraction_methods": pattern.extraction_methods,
                "times_used": pattern.times_used,
                "success_rate": pattern.success_rate,
                "created_at": pattern.created_at.isoformat(),
                "last_used": pattern.last_used.isoformat() if pattern.last_used else None,
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.log(f"Saved {len(data)} patterns to {path}", "debug")
        
        return path
    
    def load_patterns(self, path: Optional[str] = None) -> int:
        """
        Load patterns from disk.
        
        Returns the number of patterns loaded.
        """
        if path is None:
            path = os.path.join(self.config.knowledge_graph_path, "patterns.json")
        
        if not os.path.exists(path):
            return 0
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for domain, pattern_data in data.items():
            pattern = LearnedPattern(
                id=pattern_data["id"],
                site_domain=pattern_data["site_domain"],
                page_type=PageType(pattern_data["page_type"]),
                selectors=pattern_data.get("selectors", {}),
                backup_selectors=pattern_data.get("backup_selectors", {}),
                extraction_methods=pattern_data.get("extraction_methods", {}),
                times_used=pattern_data.get("times_used", 0),
                success_rate=pattern_data.get("success_rate", 0.0),
            )
            self._patterns[domain] = pattern
        
        self.log(f"Loaded {len(data)} patterns from {path}", "debug")
        
        return len(data)
    
    def generate_playwright_template(
        self,
        pattern: LearnedPattern,
    ) -> str:
        """
        Generate a Playwright extraction script from a pattern.
        
        Returns Python code as a string.
        """
        return pattern.to_playwright_code()
    
    def export_template_file(
        self,
        pattern: LearnedPattern,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export a pattern as a Playwright template file.
        
        Returns the path to the created file.
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.templates_dir,
                f"{pattern.site_domain.replace('.', '_')}_extractor.py"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        code = f'''"""
Auto-generated Playwright extraction template.
Site: {pattern.site_domain}
Generated: {datetime.now().isoformat()}
Success rate: {pattern.success_rate:.0%}

Usage:
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://{pattern.site_domain}/faculty/someone")
        data = await extract(page)
        print(data)
"""

from playwright.async_api import Page
from typing import Any, Dict, List, Optional


{pattern.to_playwright_code()}


async def extract_batch(page: Page, urls: List[str]) -> List[Dict[str, Any]]:
    """Extract data from multiple URLs."""
    results = []
    
    for url in urls:
        try:
            await page.goto(url, wait_until='networkidle', timeout=30000)
            data = await extract(page)
            data['url'] = url
            results.append(data)
        except Exception as e:
            results.append({{"url": url, "error": str(e)}})
    
    return results
'''
        
        with open(output_path, 'w') as f:
            f.write(code)
        
        self.log(f"Exported template to {output_path}", "debug")
        
        return output_path
