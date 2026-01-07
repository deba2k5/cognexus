"""
AWE Executor Agent
==================
Executes browser actions using Playwright.

The Executor:
1. Receives action plans from the Planner
2. Executes actions (navigate, click, scroll, type)
3. Handles action failures and retries
4. Reports results back
"""

from __future__ import annotations
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urljoin

if TYPE_CHECKING:
    from playwright.async_api import Page, Browser

from .base import BaseAgent, AgentCapability
from ..core.types import (
    Action,
    ActionType,
    AgentContext,
    AgentRole,
)
from ..core.config import AWEConfig


logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of executing an action."""
    action: Action
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: int = 0


class ExecutorAgent(BaseAgent):
    """
    Playwright action execution agent.
    
    Responsibilities:
    - Execute browser actions
    - Handle navigation and waiting
    - Extract data from pages
    - Report action results
    """
    
    def __init__(
        self,
        config: AWEConfig,
        browser: Optional["Browser"] = None,
        page: Optional["Page"] = None,
        state=None,
    ):
        super().__init__(AgentRole.EXECUTOR, config, state=state)
        self.browser = browser
        self.page = page
        
        self.register_capability(AgentCapability(
            name="action_execution",
            description="Execute browser actions (navigate, click, scroll, type, extract)",
            required_inputs=["actions"],
            outputs=["results"],
        ))
    
    def set_browser(self, browser: "Browser") -> None:
        """Set the browser instance."""
        self.browser = browser
    
    def set_page(self, page: "Page") -> None:
        """Set the page instance."""
        self.page = page
    
    async def process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Execute actions from context.
        """
        # Get actions from context (set by orchestrator)
        actions = getattr(context, '_pending_actions', [])
        if not actions:
            return {"error": "No actions to execute"}
        
        results = await self.execute_actions(actions)
        
        return {
            "results": results,
            "success_count": sum(1 for r in results if r.success),
            "failure_count": sum(1 for r in results if not r.success),
        }
    
    async def execute_actions(
        self,
        actions: List[Action],
        stop_on_failure: bool = False,
    ) -> List[ActionResult]:
        """
        Execute a list of actions.
        
        Args:
            actions: List of actions to execute
            stop_on_failure: Whether to stop if an action fails
        
        Returns:
            List of ActionResult objects
        """
        if not self.page:
            raise ValueError("No page available for execution")
        
        results = []
        
        for action in actions:
            result = await self.execute_action(action)
            results.append(result)
            
            action.executed = True
            action.success = result.success
            action.error = result.error
            action.result_data = result.data
            
            if not result.success and stop_on_failure:
                self.log(f"Stopping execution due to failure: {result.error}", "warning")
                break
            
            # Wait after action if specified
            if action.wait_after > 0:
                await self.page.wait_for_timeout(action.wait_after)
        
        return results
    
    async def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single action.
        
        Args:
            action: The action to execute
        
        Returns:
            ActionResult with success/failure and data
        """
        import time
        start_time = time.time()
        
        self.log(f"Executing: {action.type.value} -> {action.target or action.value}", "debug")
        
        try:
            if action.type == ActionType.NAVIGATE:
                data = await self._execute_navigate(action)
            elif action.type == ActionType.CLICK:
                data = await self._execute_click(action)
            elif action.type == ActionType.SCROLL:
                data = await self._execute_scroll(action)
            elif action.type == ActionType.TYPE:
                data = await self._execute_type(action)
            elif action.type == ActionType.WAIT:
                data = await self._execute_wait(action)
            elif action.type == ActionType.EXTRACT:
                data = await self._execute_extract(action)
            elif action.type == ActionType.SCREENSHOT:
                data = await self._execute_screenshot(action)
            elif action.type == ActionType.HOVER:
                data = await self._execute_hover(action)
            elif action.type == ActionType.SELECT:
                data = await self._execute_select(action)
            else:
                return ActionResult(
                    action=action,
                    success=False,
                    error=f"Unknown action type: {action.type}",
                )
            
            duration = int((time.time() - start_time) * 1000)
            
            return ActionResult(
                action=action,
                success=True,
                data=data,
                duration_ms=duration,
            )
            
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            self.log(f"Action failed: {e}", "warning")
            
            return ActionResult(
                action=action,
                success=False,
                error=str(e),
                duration_ms=duration,
            )
    
    async def _execute_navigate(self, action: Action) -> Dict[str, Any]:
        """Navigate to a URL."""
        url = action.target
        if not url:
            raise ValueError("Navigate action requires target URL")
        
        await self.page.goto(url, wait_until='networkidle', timeout=action.timeout)
        
        return {
            "url": self.page.url,
            "title": await self.page.title(),
        }
    
    async def _execute_click(self, action: Action) -> Dict[str, Any]:
        """Click on an element."""
        selector = action.target
        if not selector:
            raise ValueError("Click action requires target selector")
        
        await self.page.click(selector, timeout=action.timeout)
        
        return {"clicked": selector}
    
    async def _execute_scroll(self, action: Action) -> Dict[str, Any]:
        """Scroll the page."""
        target = action.target or "bottom"
        
        if target == "bottom":
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif target == "top":
            await self.page.evaluate("window.scrollTo(0, 0)")
        elif target.isdigit():
            await self.page.evaluate(f"window.scrollBy(0, {target})")
        else:
            # Scroll to element
            await self.page.locator(target).scroll_into_view_if_needed()
        
        return {"scrolled": target}
    
    async def _execute_type(self, action: Action) -> Dict[str, Any]:
        """Type text into an input."""
        selector = action.target
        text = action.value
        
        if not selector or text is None:
            raise ValueError("Type action requires target and value")
        
        await self.page.fill(selector, text, timeout=action.timeout)
        
        return {"typed": text, "into": selector}
    
    async def _execute_wait(self, action: Action) -> Dict[str, Any]:
        """Wait for a duration or element."""
        target = action.value or action.target
        
        if target and target.isdigit():
            await self.page.wait_for_timeout(int(target))
            return {"waited_ms": int(target)}
        elif target:
            await self.page.wait_for_selector(target, timeout=action.timeout)
            return {"waited_for": target}
        else:
            await self.page.wait_for_timeout(1000)
            return {"waited_ms": 1000}
    
    async def _execute_extract(self, action: Action) -> Dict[str, Any]:
        """Extract data from the page."""
        target = action.target or "all"
        
        if target == "json":
            # Extract JSON content
            return await self._extract_json()
        elif target == "profile_links":
            # Extract profile/faculty links
            return await self._extract_profile_links()
        elif target == "all":
            # Extract all links
            return await self._extract_all_links()
        else:
            # Extract using selector
            return await self._extract_with_selector(target)
    
    async def _execute_screenshot(self, action: Action) -> Dict[str, Any]:
        """Take a screenshot."""
        import base64
        
        screenshot_bytes = await self.page.screenshot(full_page=action.value == "full")
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        return {"screenshot": screenshot_base64}
    
    async def _execute_hover(self, action: Action) -> Dict[str, Any]:
        """Hover over an element."""
        selector = action.target
        if not selector:
            raise ValueError("Hover action requires target selector")
        
        await self.page.hover(selector, timeout=action.timeout)
        
        return {"hovered": selector}
    
    async def _execute_select(self, action: Action) -> Dict[str, Any]:
        """Select an option from a dropdown."""
        selector = action.target
        value = action.value
        
        if not selector or not value:
            raise ValueError("Select action requires target and value")
        
        await self.page.select_option(selector, value, timeout=action.timeout)
        
        return {"selected": value, "from": selector}
    
    # =========================================================================
    # Extraction Methods
    # =========================================================================
    
    async def _extract_json(self) -> Dict[str, Any]:
        """Extract JSON content from page."""
        content = await self.page.content()
        
        # Remove HTML tags if present
        if content.startswith('<'):
            # Try to find JSON in pre tag
            pre_match = re.search(r'<pre[^>]*>(.*?)</pre>', content, re.DOTALL)
            if pre_match:
                content = pre_match.group(1)
            else:
                # Try body
                body_match = re.search(r'<body[^>]*>(.*?)</body>', content, re.DOTALL)
                if body_match:
                    content = body_match.group(1)
        
        # Unescape common patterns
        content = content.replace('\\/', '/').replace('\\"', '"')
        
        # Extract URLs
        urls = set()
        
        # Faculty URLs
        for match in re.finditer(r'/faculty/([a-z][a-z0-9-]+)', content, re.I):
            full_url = urljoin(self.page.url, f"/faculty/{match.group(1)}")
            urls.add(full_url)
        
        # People URLs
        for match in re.finditer(r'/people/([a-z0-9-]+)', content, re.I):
            full_url = urljoin(self.page.url, f"/people/{match.group(1)}")
            urls.add(full_url)
        
        # Profile URLs
        for match in re.finditer(r'/profile/([a-z0-9-]+)', content, re.I):
            full_url = urljoin(self.page.url, f"/profile/{match.group(1)}")
            urls.add(full_url)
        
        return {
            "type": "json",
            "urls": list(urls),
            "count": len(urls),
        }
    
    async def _extract_profile_links(self) -> Dict[str, Any]:
        """Extract profile/faculty links from page."""
        links = await self.page.evaluate('''() => {
            const results = [];
            const profilePatterns = [
                /\\/faculty\\/[a-z][a-z0-9-]+$/i,
                /\\/people\\/[a-z0-9-]+$/i,
                /\\/profile\\/[a-z0-9-]+$/i,
                /\\/staff\\/[a-z][a-z0-9-]+$/i,
            ];
            
            document.querySelectorAll('a[href]').forEach(a => {
                const href = a.href;
                const isProfile = profilePatterns.some(p => p.test(href));
                
                if (isProfile) {
                    results.push({
                        url: href,
                        text: a.innerText.trim().substring(0, 100),
                    });
                }
            });
            
            // Deduplicate by URL
            const seen = new Set();
            return results.filter(r => {
                if (seen.has(r.url)) return false;
                seen.add(r.url);
                return true;
            });
        }''')
        
        return {
            "type": "profile_links",
            "links": links,
            "count": len(links),
        }
    
    async def _extract_all_links(self) -> Dict[str, Any]:
        """Extract all links from page."""
        links = await self.page.evaluate('''() => {
            const results = [];
            
            document.querySelectorAll('a[href]').forEach(a => {
                results.push({
                    url: a.href,
                    text: a.innerText.trim().substring(0, 100),
                });
            });
            
            return results;
        }''')
        
        return {
            "type": "all_links",
            "links": links,
            "count": len(links),
        }
    
    async def _extract_with_selector(self, selector: str) -> Dict[str, Any]:
        """Extract data using a CSS selector."""
        elements = await self.page.query_selector_all(selector)
        
        results = []
        for elem in elements:
            text = await elem.inner_text()
            href = await elem.get_attribute('href')
            
            results.append({
                "text": text.strip()[:200] if text else "",
                "href": href,
            })
        
        return {
            "type": "selector",
            "selector": selector,
            "elements": results,
            "count": len(results),
        }
    
    # =========================================================================
    # High-Level Methods
    # =========================================================================
    
    async def navigate_and_wait(
        self,
        url: str,
        wait_selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Navigate to URL and optionally wait for a selector.
        
        Convenience method for common navigation pattern.
        """
        actions = [
            Action(type=ActionType.NAVIGATE, target=url),
        ]
        
        if wait_selector:
            actions.append(Action(
                type=ActionType.WAIT,
                value=wait_selector,
                timeout=self.config.timeout,
            ))
        
        results = await self.execute_actions(actions)
        
        return {
            "success": all(r.success for r in results),
            "url": self.page.url,
            "results": results,
        }
    
    async def scroll_to_bottom(
        self,
        max_scrolls: int = 10,
        scroll_delay: int = 1000,
    ) -> Dict[str, Any]:
        """
        Scroll to bottom repeatedly for infinite scroll pages.
        
        Returns when no new content loads or max scrolls reached.
        """
        actions = []
        previous_height = 0
        scroll_count = 0
        
        for _ in range(max_scrolls):
            # Get current height
            current_height = await self.page.evaluate("document.body.scrollHeight")
            
            if current_height == previous_height:
                break  # No new content
            
            previous_height = current_height
            
            # Scroll
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await self.page.wait_for_timeout(scroll_delay)
            scroll_count += 1
        
        return {
            "scroll_count": scroll_count,
            "final_height": previous_height,
        }
