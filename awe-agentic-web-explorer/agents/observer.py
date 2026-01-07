"""
AWE Observer Agent
==================
Responsible for understanding pages through vision and DOM analysis.

The Observer:
1. Takes screenshots at key moments
2. Analyzes page structure (cards, lists, tables)
3. Identifies AJAX endpoints, pagination patterns
4. Classifies page type (directory, profile, search, etc.)
5. Extracts semantic DOM representation
"""

from __future__ import annotations
import asyncio
import base64
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from playwright.async_api import Page

from .base import BaseAgent, AgentCapability
from ..core.types import (
    AgentContext,
    AgentRole,
    ContentLoadingType,
    DOMElement,
    PageObservation,
    PageType,
)
from ..core.config import AWEConfig


logger = logging.getLogger(__name__)


# =============================================================================
# DOM Analysis Utilities
# =============================================================================

class DOMAnalyzer:
    """
    Analyzes DOM structure to extract meaningful patterns.
    
    Converts complex HTML into a compact semantic representation
    that LLMs can reason about effectively.
    """
    
    # Patterns that indicate different page types
    PROFILE_PATTERNS = [
        r'/faculty/[a-z][a-z0-9-]+$',
        r'/people/[a-z0-9-]+$',
        r'/profile/[a-z0-9-]+$',
        r'/staff/[a-z][a-z0-9-]+$',
        r'/directory/[a-z0-9-]+$',
    ]
    
    DIRECTORY_PATTERNS = [
        r'/faculty/?$',
        r'/people/?$',
        r'/directory/?$',
        r'/staff/?$',
        r'/team/?$',
    ]
    
    # Card-like container patterns
    CARD_CLASSES = [
        'card', 'faculty-card', 'profile-card', 'person-card',
        'member', 'faculty-member', 'team-member',
        'item', 'faculty-item', 'profile-item',
        'grid-item', 'list-item',
    ]
    
    # Pagination patterns
    PAGINATION_CLASSES = [
        'pagination', 'pager', 'page-numbers',
        'nav-links', 'page-nav',
    ]
    
    async def analyze(self, page: "Page", url: str) -> Dict[str, Any]:
        """
        Analyze a page and return structured analysis.
        
        Args:
            page: Playwright page object
            url: Current URL
        
        Returns:
            Dict with analysis results
        """
        analysis = {
            "url": url,
            "page_type": PageType.UNKNOWN,
            "content_loading": ContentLoadingType.STATIC,
            "elements": [],
            "patterns": {},
        }
        
        # Get HTML content
        html = await page.content()
        
        # Classify page type
        analysis["page_type"] = self._classify_page(url, html)
        
        # Detect content loading type
        analysis["content_loading"] = self._detect_loading_type(html)
        
        # Find card patterns
        analysis["patterns"]["cards"] = await self._find_card_patterns(page)
        
        # Find pagination
        analysis["patterns"]["pagination"] = await self._find_pagination(page)
        
        # Find AJAX endpoints
        analysis["patterns"]["ajax"] = self._find_ajax_endpoints(html)
        
        # Extract key elements
        analysis["elements"] = await self._extract_key_elements(page, url)
        
        # Build semantic DOM
        analysis["semantic_dom"] = await self._build_semantic_dom(page, analysis)
        
        return analysis
    
    def _classify_page(self, url: str, html: str) -> PageType:
        """Classify the page type based on URL and content."""
        url_lower = url.lower()
        
        # Check URL patterns
        for pattern in self.PROFILE_PATTERNS:
            if re.search(pattern, url_lower):
                return PageType.PROFILE
        
        for pattern in self.DIRECTORY_PATTERNS:
            if re.search(pattern, url_lower):
                return PageType.DIRECTORY
        
        # Check content indicators
        html_lower = html.lower()
        
        if 'login' in url_lower or 'signin' in url_lower or 'type="password"' in html_lower:
            return PageType.LOGIN
        
        if '<form' in html_lower and 'search' in html_lower:
            return PageType.SEARCH
        
        # Check for directory indicators in content
        if any(cls in html_lower for cls in self.CARD_CLASSES):
            if len(re.findall(r'/faculty/[a-z]', html_lower)) > 3:
                return PageType.DIRECTORY
        
        return PageType.UNKNOWN
    
    def _detect_loading_type(self, html: str) -> ContentLoadingType:
        """Detect how content is loaded on the page."""
        html_lower = html.lower()
        
        # Check for infinite scroll indicators
        if 'infinite' in html_lower or 'scroll' in html_lower:
            if 'load' in html_lower:
                return ContentLoadingType.AJAX_ON_SCROLL
        
        # Check for load more buttons
        if 'load-more' in html_lower or 'load more' in html_lower:
            return ContentLoadingType.AJAX_ON_CLICK
        
        # Check for data-src/data-url (lazy loading)
        if 'data-src=' in html_lower or 'data-url=' in html_lower:
            return ContentLoadingType.AJAX_ON_LOAD
        
        # Check for fetch/ajax calls
        if re.search(r'fetch\s*\(|\.ajax\s*\(|xmlhttprequest', html_lower):
            return ContentLoadingType.AJAX_ON_LOAD
        
        # Check for pagination URL patterns
        if 'page=' in html_lower or 'pageindex=' in html_lower:
            return ContentLoadingType.PAGINATION_URL
        
        return ContentLoadingType.STATIC
    
    async def _find_card_patterns(self, page: "Page") -> List[str]:
        """Find CSS classes that look like card containers."""
        selectors = []
        
        for cls in self.CARD_CLASSES:
            try:
                count = await page.locator(f'[class*="{cls}"]').count()
                if count >= 3:  # At least 3 items suggests a pattern
                    selectors.append(f'[class*="{cls}"]')
            except:
                continue
        
        return selectors
    
    async def _find_pagination(self, page: "Page") -> Dict[str, Any]:
        """Find pagination elements."""
        result = {
            "type": None,
            "selector": None,
            "total_pages": 1,
            "current_page": 1,
        }
        
        for cls in self.PAGINATION_CLASSES:
            try:
                elem = await page.locator(f'[class*="{cls}"]').first
                if await elem.count():
                    result["type"] = "numbered"
                    result["selector"] = f'[class*="{cls}"]'
                    
                    # Try to count pages
                    links = await elem.locator('a').all()
                    page_numbers = []
                    for link in links:
                        text = await link.text_content()
                        if text and text.strip().isdigit():
                            page_numbers.append(int(text.strip()))
                    
                    if page_numbers:
                        result["total_pages"] = max(page_numbers)
                    
                    break
            except:
                continue
        
        # Check for next button
        if not result["type"]:
            try:
                next_btn = await page.locator('a:has-text("Next"), button:has-text("Next")').first
                if await next_btn.count():
                    result["type"] = "next_button"
                    result["selector"] = 'a:has-text("Next"), button:has-text("Next")'
            except:
                pass
        
        return result
    
    def _find_ajax_endpoints(self, html: str) -> List[str]:
        """Find potential AJAX endpoints in the HTML."""
        endpoints = []
        
        # data-src attributes
        for match in re.finditer(r'data-src=["\']([^"\']+)["\']', html):
            endpoints.append(match.group(1))
        
        # data-url attributes
        for match in re.finditer(r'data-url=["\']([^"\']+)["\']', html):
            endpoints.append(match.group(1))
        
        # API/PHP endpoints in scripts
        for match in re.finditer(r'["\']([^"\']*(?:api|results|faculty|load)[^"\']*\.php[^"\']*)["\']', html, re.I):
            endpoints.append(match.group(1))
        
        # JSON endpoints
        for match in re.finditer(r'["\']([^"\']+\.json)["\']', html):
            endpoints.append(match.group(1))
        
        # Umbraco-style endpoints
        for match in re.finditer(r'/umbraco/[^"\'<>\s]+', html):
            endpoints.append(match.group(0))
        
        return list(set(endpoints))
    
    async def _extract_key_elements(self, page: "Page", base_url: str) -> List[DOMElement]:
        """Extract key elements from the page."""
        elements = []
        
        # Extract profile links
        links = await page.evaluate('''(baseUrl) => {
            const results = [];
            const profilePatterns = [
                /\\/faculty\\/[a-z][a-z0-9-_]+/i,
                /\\/people\\/[a-z0-9-_]+/i,
                /\\/profile\\/[a-z0-9-_]+/i,
                /\\/~[a-z0-9]+/i,  // Unix-style profiles like /~username
                /\\/staff\\/[a-z0-9-_]+/i,
                /\\/directory\\/[a-z0-9-_]+/i,
            ];
            
            // Common non-name words that indicate navigation
            const navWords = [
                'program', 'admissions', 'research', 'areas', 'reports', 
                'facilities', 'staff', 'students', 'alumni', 'graduate',
                'undergraduate', 'about', 'contact', 'news', 'events',
                'home', 'overview', 'technical', 'administrative', 'learning',
                'computing', 'requirements', 'directory', 'all', 'more'
            ];
            
            document.querySelectorAll('a[href]').forEach(a => {
                const href = a.href;
                const text = a.innerText.trim();
                
                // Check URL patterns
                const isProfile = profilePatterns.some(p => p.test(href));
                
                // Check if the link text looks like a person name (2-4 words, each capitalized)
                // Must not contain common navigation words
                const words = text.split(/\\s+/);
                const wordCount = words.length;
                const hasNavWord = words.some(w => navWords.includes(w.toLowerCase()));
                const allWordsCapitalized = words.every(w => /^[A-Z][a-z'-]+$/.test(w));
                const looksLikeName = wordCount >= 2 && wordCount <= 4 && 
                                      allWordsCapitalized && 
                                      !hasNavWord &&
                                      text.length > 5 && text.length < 40;
                
                if (isProfile || looksLikeName) {
                    results.push({
                        tag: 'a',
                        selector: '',
                        href: href,
                        text: text.substring(0, 100),
                        classes: Array.from(a.classList),
                        parentClasses: a.parentElement ? Array.from(a.parentElement.classList) : [],
                        isNameLink: looksLikeName,
                    });
                }
            });
            
            // Sort to prioritize name-like links
            results.sort((a, b) => (b.isNameLink ? 1 : 0) - (a.isNameLink ? 1 : 0));
            
            return results;
        }''', base_url)
        
        for link in links:
            # Only add links that look like person names
            if link.get('isNameLink', False):
                elements.append(DOMElement(
                    tag=link['tag'],
                    selector='a[href*="/faculty/"], a[href*="/people/"]',
                    classes=link.get('classes', []),
                    text=link.get('text', ''),
                    href=link.get('href'),
                    looks_like_link=True,
                ))
        
        return elements
    
    async def _build_semantic_dom(self, page: "Page", analysis: Dict) -> str:
        """
        Build a compact semantic representation of the DOM.
        
        This is what gets sent to the LLM for reasoning.
        """
        lines = []
        lines.append(f'<page url="{analysis["url"]}" type="{analysis["page_type"].value}">')
        
        # Add content loading info
        lines.append(f'  <content_loading>{analysis["content_loading"].value}</content_loading>')
        
        # Add card patterns
        if analysis["patterns"]["cards"]:
            lines.append(f'  <cards_found patterns="{len(analysis["patterns"]["cards"])}">')
            for pattern in analysis["patterns"]["cards"][:3]:
                count = await page.locator(pattern).count()
                lines.append(f'    <pattern selector="{pattern}" count="{count}"/>')
            lines.append('  </cards_found>')
        
        # Add pagination
        pag = analysis["patterns"]["pagination"]
        if pag["type"]:
            lines.append(f'  <pagination type="{pag["type"]}" total="{pag["total_pages"]}" current="{pag["current_page"]}"/>')
        
        # Add AJAX endpoints
        if analysis["patterns"]["ajax"]:
            lines.append('  <ajax_endpoints>')
            for endpoint in analysis["patterns"]["ajax"][:5]:
                lines.append(f'    <endpoint>{endpoint}</endpoint>')
            lines.append('  </ajax_endpoints>')
        
        # Add element summary
        profile_links = [e for e in analysis["elements"] if e.looks_like_link]
        lines.append(f'  <profile_links count="{len(profile_links)}">')
        for elem in profile_links[:5]:
            lines.append(f'    <link href="{elem.href}" text="{elem.text[:30]}"/>')
        if len(profile_links) > 5:
            lines.append(f'    <!-- and {len(profile_links) - 5} more -->')
        lines.append('  </profile_links>')
        
        lines.append('</page>')
        
        return '\n'.join(lines)


# =============================================================================
# Observer Agent
# =============================================================================

class ObserverAgent(BaseAgent):
    """
    Vision and DOM analysis agent.
    
    Responsibilities:
    - Take screenshots for visual understanding
    - Analyze page structure
    - Classify page types
    - Detect patterns (pagination, AJAX, cards)
    - Build semantic DOM representation
    """
    
    VISION_PROMPT = """Analyze this web page screenshot and answer:

1. What type of page is this? (faculty directory, profile page, search interface, login, other)
2. How is content organized? (grid of cards, list, table, single article)
3. Are there pagination controls visible? If so, what type? (numbered, next/prev, load more)
4. Approximately how many items are visible on the page?
5. Is there a search or filter interface?
6. What are the main interactive elements?

Be concise and specific. Focus on what's useful for web scraping."""

    def __init__(
        self,
        config: AWEConfig,
        llm_func: Optional[Callable] = None,
        vision_func: Optional[Callable] = None,
        state=None,
    ):
        super().__init__(AgentRole.OBSERVER, config, llm_func, state)
        self.vision_llm = vision_func or llm_func
        self.dom_analyzer = DOMAnalyzer()
        
        self.register_capability(AgentCapability(
            name="page_observation",
            description="Observe and analyze web pages using vision and DOM analysis",
            required_inputs=["page"],
            outputs=["observation"],
        ))
    
    async def process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Process a page and return a complete observation.
        """
        if not self.state:
            return {"error": "No state store available"}
        
        observation = await self.state.get_latest_observation()
        if observation:
            return {"observation": observation}
        
        return {"error": "No observation available"}
    
    async def observe(
        self,
        page: "Page",
        url: str,
        take_screenshot: bool = True,
    ) -> PageObservation:
        """
        Create a complete observation of the current page.
        
        Args:
            page: Playwright page object
            url: Current URL
            take_screenshot: Whether to capture a screenshot
        
        Returns:
            PageObservation with all analysis data
        """
        self.log(f"Observing: {url}", "debug")
        
        # Get basic info
        title = await page.title()
        html = await page.content()
        
        # Take screenshot if enabled
        screenshot_base64 = None
        if take_screenshot and self.config.screenshots_enabled:
            screenshot_bytes = await page.screenshot(full_page=False)
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # DOM analysis
        dom_analysis = await self.dom_analyzer.analyze(page, url)
        
        # Vision analysis if available
        vision_analysis = {}
        if screenshot_base64 and self.vision_llm and self.config.vision_enabled:
            try:
                vision_analysis = await self._analyze_with_vision(screenshot_base64)
            except Exception as e:
                self.log(f"Vision analysis failed: {e}", "warning")
        
        # Combine analyses
        page_type = dom_analysis["page_type"]
        if vision_analysis.get("page_type") and page_type == PageType.UNKNOWN:
            page_type = PageType(vision_analysis["page_type"])
        
        observation = PageObservation(
            url=url,
            title=title,
            timestamp=datetime.now(),
            screenshot_base64=screenshot_base64,
            html=html,
            semantic_dom=dom_analysis["semantic_dom"],
            elements=dom_analysis["elements"],
            page_type=page_type,
            content_loading=dom_analysis["content_loading"],
            link_patterns=[],
            card_patterns=dom_analysis["patterns"]["cards"],
            pagination_info=dom_analysis["patterns"]["pagination"],
            ajax_endpoints=dom_analysis["patterns"]["ajax"],
            total_links=len(dom_analysis["elements"]),
            profile_links=len([e for e in dom_analysis["elements"] if e.looks_like_link]),
            visible_items=vision_analysis.get("visible_items", len(dom_analysis["patterns"]["cards"])),
        )
        
        # Save to state
        if self.state:
            await self.state.save_observation(observation)
        
        self.log(f"Observation complete: type={page_type.value}, links={observation.profile_links}", "debug")
        
        return observation
    
    async def _analyze_with_vision(self, screenshot_base64: str) -> Dict[str, Any]:
        """Analyze screenshot using vision LLM."""
        try:
            # Call vision LLM with screenshot
            response = await self.vision_llm(
                self.VISION_PROMPT,
                image_base64=screenshot_base64,
            )
            
            # Parse response
            analysis = self._parse_vision_response(response)
            return analysis
            
        except Exception as e:
            self.log(f"Vision analysis error: {e}", "warning")
            return {}
    
    def _parse_vision_response(self, response: str) -> Dict[str, Any]:
        """Parse vision LLM response into structured data."""
        analysis = {}
        response_lower = response.lower()
        
        # Page type
        if 'directory' in response_lower or 'list of' in response_lower:
            analysis["page_type"] = PageType.DIRECTORY.value
        elif 'profile' in response_lower or 'individual' in response_lower:
            analysis["page_type"] = PageType.PROFILE.value
        elif 'search' in response_lower:
            analysis["page_type"] = PageType.SEARCH.value
        
        # Visible items count (try to extract number)
        import re
        numbers = re.findall(r'(\d+)\s*(?:items?|profiles?|cards?|people|faculty)', response_lower)
        if numbers:
            analysis["visible_items"] = int(numbers[0])
        
        return analysis
    
    async def observe_multiple(
        self,
        page: "Page",
        urls: List[str],
    ) -> List[PageObservation]:
        """
        Observe multiple pages.
        
        Useful for batch observation of profile pages.
        """
        observations = []
        
        for url in urls:
            try:
                await page.goto(url, wait_until='networkidle', timeout=self.config.timeout)
                await page.wait_for_timeout(500)
                observation = await self.observe(page, url)
                observations.append(observation)
            except Exception as e:
                self.log(f"Failed to observe {url}: {e}", "warning")
        
        return observations
