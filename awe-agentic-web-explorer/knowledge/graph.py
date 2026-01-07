"""
AWE Knowledge Graph
====================

Persistent storage and retrieval for learned patterns and exploration knowledge.

The Knowledge Graph:
1. Stores learned patterns across sessions
2. Enables similarity-based pattern lookup
3. Tracks site structure and navigation paths
4. Builds domain expertise over time
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from ..core.types import (
    LearnedPattern,
    PageType,
    ExtractionResult,
)


@dataclass
class SiteNode:
    """A node in the site knowledge graph."""
    url_pattern: str  # e.g., "/faculty/{id}"
    page_type: PageType
    selectors: Dict[str, str]
    links_to: Set[str] = field(default_factory=set)  # URL patterns this links to
    extraction_fields: List[str] = field(default_factory=list)
    success_count: int = 0
    fail_count: int = 0
    last_visited: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url_pattern": self.url_pattern,
            "page_type": self.page_type.value if hasattr(self.page_type, "value") else str(self.page_type),
            "selectors": self.selectors,
            "links_to": list(self.links_to),
            "extraction_fields": self.extraction_fields,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "last_visited": self.last_visited,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiteNode":
        page_type = data.get("page_type", "unknown")
        if isinstance(page_type, str):
            try:
                page_type = PageType(page_type)
            except:
                page_type = PageType.UNKNOWN
        
        return cls(
            url_pattern=data["url_pattern"],
            page_type=page_type,
            selectors=data.get("selectors", {}),
            links_to=set(data.get("links_to", [])),
            extraction_fields=data.get("extraction_fields", []),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            last_visited=data.get("last_visited"),
        )


@dataclass
class DomainKnowledge:
    """Knowledge about a specific domain."""
    domain: str
    nodes: Dict[str, SiteNode] = field(default_factory=dict)  # url_pattern -> node
    patterns: Dict[str, LearnedPattern] = field(default_factory=dict)  # page_type -> pattern
    navigation_paths: List[List[str]] = field(default_factory=list)  # Successful nav paths
    total_extractions: int = 0
    successful_extractions: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "patterns": {k: {
                "domain": v.domain,
                "page_type": v.page_type.value if hasattr(v.page_type, "value") else str(v.page_type),
                "selectors": v.selectors,
                "usage_count": v.usage_count,
                "success_rate": v.success_rate,
            } for k, v in self.patterns.items()},
            "navigation_paths": self.navigation_paths,
            "total_extractions": self.total_extractions,
            "successful_extractions": self.successful_extractions,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainKnowledge":
        nodes = {}
        for k, v in data.get("nodes", {}).items():
            nodes[k] = SiteNode.from_dict(v)
        
        patterns = {}
        for k, v in data.get("patterns", {}).items():
            page_type = v.get("page_type", "unknown")
            if isinstance(page_type, str):
                try:
                    page_type = PageType(page_type)
                except:
                    page_type = PageType.UNKNOWN
            
            patterns[k] = LearnedPattern(
                domain=v["domain"],
                page_type=page_type,
                selectors=v.get("selectors", {}),
                usage_count=v.get("usage_count", 0),
                success_rate=v.get("success_rate", 0.0),
            )
        
        return cls(
            domain=data["domain"],
            nodes=nodes,
            patterns=patterns,
            navigation_paths=data.get("navigation_paths", []),
            total_extractions=data.get("total_extractions", 0),
            successful_extractions=data.get("successful_extractions", 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class KnowledgeGraph:
    """
    Persistent knowledge storage for web exploration.
    
    Stores:
    - Site structure (URL patterns, page types)
    - Learned selectors and extraction patterns
    - Successful navigation paths
    - Domain-specific expertise
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the knowledge graph.
        
        Args:
            storage_path: Directory for persistent storage
        """
        self.storage_path = storage_path or Path.home() / ".awe" / "knowledge"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._domains: Dict[str, DomainKnowledge] = {}
        self._load_all()
    
    def _load_all(self):
        """Load all domain knowledge from disk."""
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    domain = data.get("domain")
                    if domain:
                        self._domains[domain] = DomainKnowledge.from_dict(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def _save_domain(self, domain: str):
        """Save domain knowledge to disk."""
        if domain not in self._domains:
            return
        
        knowledge = self._domains[domain]
        knowledge.updated_at = datetime.now().isoformat()
        
        filename = domain.replace(".", "_") + ".json"
        filepath = self.storage_path / filename
        
        with open(filepath, "w") as f:
            json.dump(knowledge.to_dict(), f, indent=2)
    
    def get_domain(self, url: str) -> DomainKnowledge:
        """Get or create knowledge for a domain."""
        domain = urlparse(url).netloc
        
        if domain not in self._domains:
            self._domains[domain] = DomainKnowledge(
                domain=domain,
                created_at=datetime.now().isoformat(),
            )
        
        return self._domains[domain]
    
    def add_node(
        self,
        url: str,
        page_type: PageType,
        selectors: Dict[str, str],
        extraction_fields: List[str] = None,
    ) -> SiteNode:
        """Add or update a site node."""
        domain_knowledge = self.get_domain(url)
        url_pattern = self._url_to_pattern(url)
        
        if url_pattern in domain_knowledge.nodes:
            node = domain_knowledge.nodes[url_pattern]
            node.selectors.update(selectors)
            if extraction_fields:
                node.extraction_fields = list(set(node.extraction_fields + extraction_fields))
        else:
            node = SiteNode(
                url_pattern=url_pattern,
                page_type=page_type,
                selectors=selectors,
                extraction_fields=extraction_fields or [],
            )
            domain_knowledge.nodes[url_pattern] = node
        
        node.last_visited = datetime.now().isoformat()
        self._save_domain(urlparse(url).netloc)
        
        return node
    
    def add_link(self, from_url: str, to_url: str):
        """Record a link between pages."""
        domain_knowledge = self.get_domain(from_url)
        
        from_pattern = self._url_to_pattern(from_url)
        to_pattern = self._url_to_pattern(to_url)
        
        if from_pattern in domain_knowledge.nodes:
            domain_knowledge.nodes[from_pattern].links_to.add(to_pattern)
            self._save_domain(urlparse(from_url).netloc)
    
    def record_extraction(
        self,
        url: str,
        success: bool,
        pattern: Optional[LearnedPattern] = None,
    ):
        """Record an extraction attempt."""
        domain_knowledge = self.get_domain(url)
        url_pattern = self._url_to_pattern(url)
        
        domain_knowledge.total_extractions += 1
        if success:
            domain_knowledge.successful_extractions += 1
        
        if url_pattern in domain_knowledge.nodes:
            node = domain_knowledge.nodes[url_pattern]
            if success:
                node.success_count += 1
            else:
                node.fail_count += 1
        
        if pattern:
            key = pattern.page_type.value if hasattr(pattern.page_type, "value") else str(pattern.page_type)
            domain_knowledge.patterns[key] = pattern
        
        self._save_domain(urlparse(url).netloc)
    
    def record_navigation_path(self, urls: List[str]):
        """Record a successful navigation path."""
        if not urls:
            return
        
        domain_knowledge = self.get_domain(urls[0])
        patterns = [self._url_to_pattern(u) for u in urls]
        
        if patterns not in domain_knowledge.navigation_paths:
            domain_knowledge.navigation_paths.append(patterns)
            self._save_domain(urlparse(urls[0]).netloc)
    
    def get_pattern(self, url: str, page_type: PageType) -> Optional[LearnedPattern]:
        """Get learned pattern for a page type."""
        domain_knowledge = self.get_domain(url)
        key = page_type.value if hasattr(page_type, "value") else str(page_type)
        return domain_knowledge.patterns.get(key)
    
    def get_selectors(self, url: str) -> Dict[str, str]:
        """Get known selectors for a URL."""
        domain_knowledge = self.get_domain(url)
        url_pattern = self._url_to_pattern(url)
        
        if url_pattern in domain_knowledge.nodes:
            return domain_knowledge.nodes[url_pattern].selectors
        
        return {}
    
    def get_site_structure(self, url: str) -> Dict[str, Any]:
        """Get the known site structure for a domain."""
        domain_knowledge = self.get_domain(url)
        
        return {
            "domain": domain_knowledge.domain,
            "total_pages": len(domain_knowledge.nodes),
            "page_types": list(set(
                n.page_type.value if hasattr(n.page_type, "value") else str(n.page_type)
                for n in domain_knowledge.nodes.values()
            )),
            "success_rate": (
                domain_knowledge.successful_extractions / domain_knowledge.total_extractions
                if domain_knowledge.total_extractions > 0 else 0.0
            ),
            "navigation_paths": domain_knowledge.navigation_paths[:5],  # Top 5 paths
        }
    
    def find_similar_domain(self, url: str) -> Optional[DomainKnowledge]:
        """Find a similar domain that might have transferable patterns."""
        target_domain = urlparse(url).netloc
        
        # Look for same TLD pattern (e.g., .edu sites)
        target_parts = target_domain.split(".")
        target_tld = ".".join(target_parts[-2:]) if len(target_parts) > 1 else target_parts[0]
        
        best_match = None
        best_score = 0
        
        for domain, knowledge in self._domains.items():
            if domain == target_domain:
                continue
            
            parts = domain.split(".")
            tld = ".".join(parts[-2:]) if len(parts) > 1 else parts[0]
            
            score = 0
            
            # Same TLD bonus
            if tld == target_tld:
                score += 10
            
            # Has successful patterns
            if knowledge.patterns:
                score += 5 * len(knowledge.patterns)
            
            # High success rate
            if knowledge.total_extractions > 0:
                rate = knowledge.successful_extractions / knowledge.total_extractions
                score += rate * 20
            
            if score > best_score:
                best_score = score
                best_match = knowledge
        
        return best_match
    
    def export_knowledge(self, url: str) -> Dict[str, Any]:
        """Export all knowledge for a domain."""
        domain_knowledge = self.get_domain(url)
        return domain_knowledge.to_dict()
    
    def import_knowledge(self, data: Dict[str, Any]):
        """Import knowledge from external source."""
        domain = data.get("domain")
        if domain:
            self._domains[domain] = DomainKnowledge.from_dict(data)
            self._save_domain(domain)
    
    def _url_to_pattern(self, url: str) -> str:
        """Convert a URL to a pattern."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        
        # Simple pattern extraction - replace IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
            flags=re.IGNORECASE,
        )
        
        # Replace numeric IDs
        path = re.sub(r"/\d+(/|$)", "/{id}\\1", path)
        
        # Replace encoded names (likely slugs)
        path = re.sub(r"/[A-Z][a-z]+[A-Z][a-zA-Z]+\.html", "/{name}.html", path)
        
        return path or "/"
    
    def clear_domain(self, url: str):
        """Clear all knowledge for a domain."""
        domain = urlparse(url).netloc
        
        if domain in self._domains:
            del self._domains[domain]
            
            filename = domain.replace(".", "_") + ".json"
            filepath = self.storage_path / filename
            if filepath.exists():
                filepath.unlink()
    
    def stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        total_domains = len(self._domains)
        total_nodes = sum(len(d.nodes) for d in self._domains.values())
        total_patterns = sum(len(d.patterns) for d in self._domains.values())
        total_extractions = sum(d.total_extractions for d in self._domains.values())
        successful = sum(d.successful_extractions for d in self._domains.values())
        
        return {
            "domains": total_domains,
            "nodes": total_nodes,
            "patterns": total_patterns,
            "total_extractions": total_extractions,
            "successful_extractions": successful,
            "success_rate": successful / total_extractions if total_extractions > 0 else 0.0,
        }
