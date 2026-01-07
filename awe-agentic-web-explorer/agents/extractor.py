"""
AWE Extractor Agent
===================
Extracts structured data from pages using CSS selectors and LLM fallback.

The Extractor:
1. Uses learned patterns when available
2. Falls back to heuristic extraction
3. Uses LLM for complex/unknown pages
4. Validates extracted data
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urljoin

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


# =============================================================================
# Extraction Utilities
# =============================================================================

class HeuristicExtractor:
    """
    Extracts data using common CSS selector patterns.
    No LLM required - works based on DOM analysis.
    """
    
    # Common selectors for each field
    FIELD_SELECTORS = {
        "name": [
            "h1.masthead__title",
            "h1.faculty-name",
            "h1.profile-name",
            ".name h1",
            "h1.name",
            "h1",
        ],
        "title": [
            "h2.masthead__subtitle span",
            "h2.masthead__subtitle",
            ".faculty-title",
            ".position",
            ".title",
            ".role",
            ".designation",
        ],
        "email": [
            "a[href^='mailto:']",
        ],
        "phone": [
            "a[href^='tel:']",
        ],
        "department": [
            ".department",
            ".dept",
            "[class*='department']",
            ".affiliation",
            ".school",
        ],
        "bio": [
            ".biography p",
            ".bio p",
            ".about p",
            ".content > p",
            "article p",
            ".description p",
        ],
        "image": [
            "img.profile",
            "img.faculty-photo",
            ".profile-photo img",
            ".headshot img",
            "img[class*='profile']",
            "img[class*='photo']",
        ],
        "office": [
            ".office",
            ".location",
            "[class*='office']",
            "address",
        ],
    }
    
    async def extract(
        self,
        page: "Page",
        fields: List[str],
        base_url: str,
    ) -> Dict[str, Any]:
        """
        Extract fields using heuristic selectors.
        
        Args:
            page: Playwright page
            fields: Fields to extract
            base_url: Base URL for resolving relative URLs
        
        Returns:
            Dict of field -> value
        """
        data = {}
        
        for field_name in fields:
            selectors = self.FIELD_SELECTORS.get(field_name, [])
            
            for selector in selectors:
                try:
                    value = await self._extract_field(page, selector, field_name, base_url)
                    if value:
                        data[field_name] = value
                        break
                except Exception as e:
                    continue
        
        # Special handling for education (list field)
        if "education" in fields:
            data["education"] = await self._extract_education(page)
        
        # Special handling for research areas (list field)
        if "research_areas" in fields:
            data["research_areas"] = await self._extract_research_areas(page)
        
        return data
    
    async def _extract_field(
        self,
        page: "Page",
        selector: str,
        field_name: str,
        base_url: str,
    ) -> Optional[str]:
        """Extract a single field using selector."""
        elem = await page.query_selector(selector)
        if not elem:
            return None
        
        if field_name == "email":
            href = await elem.get_attribute("href")
            if href:
                return href.replace("mailto:", "").split("?")[0]
        
        elif field_name == "phone":
            href = await elem.get_attribute("href")
            if href:
                return href.replace("tel:", "")
        
        elif field_name == "image":
            src = await elem.get_attribute("src")
            if src:
                if not src.startswith(("http://", "https://")):
                    src = urljoin(base_url, src)
                return src
        
        else:
            text = await elem.inner_text()
            if text:
                return text.strip()
        
        return None
    
    async def _extract_education(self, page: "Page") -> List[str]:
        """Extract education entries using JS traversal."""
        try:
            items = await page.evaluate('''() => {
                const results = [];
                const h3s = document.querySelectorAll('h3');
                
                for (const h3 of h3s) {
                    if (h3.textContent.toLowerCase().includes('education')) {
                        let el = h3.nextElementSibling;
                        while (el) {
                            if (el.tagName === 'H3') break;
                            
                            if (el.tagName === 'UL') {
                                el.querySelectorAll('li').forEach(li => {
                                    results.push(li.textContent.trim());
                                });
                                break;
                            }
                            
                            if (el.tagName === 'P') {
                                const ul = el.querySelector('ul');
                                if (ul) {
                                    ul.querySelectorAll('li').forEach(li => {
                                        results.push(li.textContent.trim());
                                    });
                                    break;
                                }
                            }
                            
                            el = el.nextElementSibling;
                        }
                    }
                }
                
                return results;
            }''')
            return items
        except:
            return []
    
    async def _extract_research_areas(self, page: "Page") -> List[str]:
        """Extract research areas."""
        try:
            items = await page.evaluate('''() => {
                const results = [];
                const h3s = document.querySelectorAll('h3');
                
                for (const h3 of h3s) {
                    const text = h3.textContent.toLowerCase();
                    if (text.includes('research') || text.includes('expertise') || text.includes('interest')) {
                        let el = h3.nextElementSibling;
                        while (el && el.tagName !== 'H3') {
                            if (el.tagName === 'P') {
                                const content = el.textContent.trim();
                                if (content && content.length > 5) {
                                    // Split by comma
                                    content.split(',').forEach(item => {
                                        const trimmed = item.trim();
                                        if (trimmed.length > 2) {
                                            results.push(trimmed);
                                        }
                                    });
                                    break;
                                }
                            }
                            el = el.nextElementSibling;
                        }
                    }
                }
                
                return results.slice(0, 10);  // Limit to 10
            }''')
            return items
        except:
            return []


# =============================================================================
# Extractor Agent
# =============================================================================

class ExtractorAgent(BaseAgent):
    """
    Data extraction agent.
    
    Responsibilities:
    - Extract data from profile pages
    - Use patterns when available
    - Fall back to heuristics and LLM
    - Validate extraction quality
    """
    
    LLM_EXTRACTION_PROMPT = """Extract faculty information from this HTML.
Return ONLY valid JSON with these fields (use null if not found):

{{
  "name": "Full name",
  "title": "Academic title/position",
  "email": "Email address",
  "phone": "Phone number",
  "department": "Department name",
  "office": "Office location",
  "education": ["Degree 1", "Degree 2"],
  "research_areas": ["Area 1", "Area 2"],
  "bio": "Short biography (max 500 chars)"
}}

HTML:
{html}

JSON:"""

    def __init__(
        self,
        config: AWEConfig,
        llm_func: Optional[Callable] = None,
        state=None,
    ):
        super().__init__(AgentRole.EXTRACTOR, config, llm_func, state)
        self.heuristic = HeuristicExtractor()
        
        self.register_capability(AgentCapability(
            name="data_extraction",
            description="Extract structured data from web pages",
            required_inputs=["page", "fields"],
            outputs=["extraction_result"],
        ))
    
    async def process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Extract data from the current observation.
        """
        if not context.observation:
            return {"error": "No observation available"}
        
        # This would be called with a page object in practice
        return {"error": "Use extract() method with page object"}
    
    async def extract(
        self,
        page: "Page",
        url: str,
        fields: Optional[List[str]] = None,
        pattern: Optional[LearnedPattern] = None,
    ) -> ExtractionResult:
        """
        Extract data from a page.
        
        Args:
            page: Playwright page object
            url: Current URL
            fields: Fields to extract (uses config default if None)
            pattern: Learned pattern to use (if available)
        
        Returns:
            ExtractionResult with extracted data
        """
        fields = fields or self.config.default_fields
        html = await page.content()
        
        self.log(f"Extracting from {url}", "debug")
        
        data = {}
        method_used = "unknown"
        
        # 1. Try learned pattern first
        if pattern:
            self.log("Using learned pattern", "debug")
            data = await self._extract_with_pattern(page, pattern, url)
            method_used = "pattern"
        
        # 2. If pattern failed or not available, try heuristics
        if not data or len(data) < 3:
            self.log("Using heuristic extraction", "debug")
            heuristic_data = await self.heuristic.extract(page, fields, url)
            data.update(heuristic_data)
            method_used = "heuristic"
        
        # 3. If still missing fields and LLM enabled, use LLM
        missing_fields = [f for f in fields if f not in data or not data[f]]
        if missing_fields and self.config.llm_extraction and self.llm:
            self.log(f"Using LLM for missing fields: {missing_fields}", "debug")
            llm_data = await self._extract_with_llm(html, missing_fields)
            data.update(llm_data)
            method_used = "llm_fallback"
        
        # Validate
        is_valid = self._validate_extraction(data, fields)
        
        result = ExtractionResult(
            url=url,
            data=data,
            raw_html=html if self.config.save_html else "",
            fields_found=len([f for f in fields if f in data and data[f]]),
            fields_expected=len(fields),
            confidence=len([f for f in fields if f in data and data[f]]) / len(fields),
            template_used=pattern.id if pattern else None,
            is_valid=is_valid,
            validation_errors=self._get_validation_errors(data, fields),
        )
        
        # Save to state
        if self.state:
            await self.state.add_result(result)
        
        return result
    
    async def _extract_with_pattern(
        self,
        page: "Page",
        pattern: LearnedPattern,
        base_url: str,
    ) -> Dict[str, Any]:
        """Extract using a learned pattern."""
        data = {}
        
        for field_name, selector in pattern.selectors.items():
            try:
                method = pattern.extraction_methods.get(field_name, "text")
                elem = await page.query_selector(selector)
                
                if not elem:
                    # Try backup selectors
                    for backup in pattern.backup_selectors.get(field_name, []):
                        elem = await page.query_selector(backup)
                        if elem:
                            break
                
                if not elem:
                    continue
                
                if method == "text":
                    value = await elem.inner_text()
                    data[field_name] = value.strip() if value else None
                elif method == "href":
                    value = await elem.get_attribute("href")
                    if value and value.startswith("mailto:"):
                        data[field_name] = value.replace("mailto:", "").split("?")[0]
                    elif value and value.startswith("tel:"):
                        data[field_name] = value.replace("tel:", "")
                    else:
                        data[field_name] = value
                elif method == "src":
                    value = await elem.get_attribute("src")
                    if value and not value.startswith(("http://", "https://")):
                        value = urljoin(base_url, value)
                    data[field_name] = value
                elif method == "list":
                    items = await page.query_selector_all(selector)
                    data[field_name] = [
                        (await item.inner_text()).strip()
                        for item in items
                    ]
                    
            except Exception as e:
                self.log(f"Pattern extraction failed for {field_name}: {e}", "warning")
        
        return data
    
    async def _extract_with_llm(
        self,
        html: str,
        fields: List[str],
    ) -> Dict[str, Any]:
        """Extract using LLM for complex cases."""
        if not self.llm:
            return {}
        
        # Truncate HTML to main content
        truncated_html = self._truncate_html(html, max_chars=6000)
        
        prompt = self.LLM_EXTRACTION_PROMPT.format(html=truncated_html)
        
        try:
            response = await self.llm(prompt)
            data = self._parse_llm_response(response)
            return data
        except Exception as e:
            self.log(f"LLM extraction failed: {e}", "warning")
            return {}
    
    def _truncate_html(self, html: str, max_chars: int = 6000) -> str:
        """Truncate HTML to main content area."""
        # Try to find main content
        import re
        
        # Look for main content areas
        patterns = [
            (r'<main[^>]*>(.*?)</main>', 1),
            (r'<article[^>]*>(.*?)</article>', 1),
            (r'<div[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</div>', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, html, re.DOTALL | re.I)
            if match:
                content = match.group(group)
                if len(content) > max_chars:
                    return content[:max_chars] + "..."
                return content
        
        # Fallback: return middle portion
        if len(html) > max_chars:
            start = len(html) // 4
            return html[start:start + max_chars] + "..."
        
        return html
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        # Find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return {}
        
        try:
            data = json.loads(json_match.group())
            # Filter out None/null values
            return {k: v for k, v in data.items() if v is not None}
        except json.JSONDecodeError:
            return {}
    
    def _validate_extraction(
        self,
        data: Dict[str, Any],
        expected_fields: List[str],
    ) -> bool:
        """Validate extracted data."""
        # Must have name at minimum
        if not data.get("name"):
            return False
        
        # Name shouldn't be too short or too long
        name = data.get("name", "")
        if len(name) < 3 or len(name) > 100:
            return False
        
        # Email should look like email
        email = data.get("email", "")
        if email and "@" not in email:
            return False
        
        # At least 30% of expected fields should be present
        found = sum(1 for f in expected_fields if f in data and data[f])
        if found / len(expected_fields) < 0.3:
            return False
        
        return True
    
    def _get_validation_errors(
        self,
        data: Dict[str, Any],
        expected_fields: List[str],
    ) -> List[str]:
        """Get list of validation errors."""
        errors = []
        
        if not data.get("name"):
            errors.append("Missing name")
        
        email = data.get("email", "")
        if email and "@" not in email:
            errors.append(f"Invalid email: {email}")
        
        missing = [f for f in expected_fields if f not in data or not data[f]]
        if missing:
            errors.append(f"Missing fields: {', '.join(missing)}")
        
        return errors
    
    async def extract_batch(
        self,
        page: "Page",
        urls: List[str],
        fields: Optional[List[str]] = None,
    ) -> List[ExtractionResult]:
        """
        Extract data from multiple pages.
        
        Args:
            page: Playwright page object
            urls: List of URLs to extract from
            fields: Fields to extract
        
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        for i, url in enumerate(urls):
            self.log(f"Extracting [{i+1}/{len(urls)}]: {url}", "debug")
            
            try:
                await page.goto(url, wait_until='networkidle', timeout=self.config.timeout)
                await page.wait_for_timeout(500)
                
                result = await self.extract(page, url, fields)
                results.append(result)
                
                # Rate limiting
                await asyncio.sleep(self.config.request_delay)
                
            except Exception as e:
                self.log(f"Failed to extract from {url}: {e}", "warning")
                results.append(ExtractionResult(
                    url=url,
                    is_valid=False,
                    validation_errors=[str(e)],
                ))
        
        return results
