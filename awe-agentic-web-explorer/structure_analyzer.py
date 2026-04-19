"""
AWE Website Structure Analyzer
=================================
Analyzes website structure: DOM depth, heading hierarchy, 
sitemap tree, link graph, and page metadata.
"""

import asyncio
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")


# =============================================================================
# DOM Analysis
# =============================================================================

def analyze_dom_depth(soup: BeautifulSoup) -> Dict[str, Any]:
    """Analyze DOM tree depth and complexity."""
    def get_depth(element, current_depth=0):
        if not hasattr(element, 'children'):
            return current_depth
        max_child_depth = current_depth
        for child in element.children:
            if hasattr(child, 'name') and child.name:
                child_depth = get_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth

    body = soup.find('body') or soup
    max_depth = get_depth(body)

    # Count elements by tag
    tag_counts = Counter(tag.name for tag in soup.find_all(True))
    total_elements = sum(tag_counts.values())

    # Top tags
    top_tags = tag_counts.most_common(15)

    return {
        "max_depth": max_depth,
        "total_elements": total_elements,
        "unique_tags": len(tag_counts),
        "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
        "complexity": "high" if max_depth > 20 or total_elements > 1000 else "medium" if max_depth > 10 else "low",
    }


# =============================================================================
# Heading Hierarchy
# =============================================================================

def analyze_headings(soup: BeautifulSoup) -> Dict[str, Any]:
    """Analyze heading hierarchy and SEO issues."""
    headings = []
    issues = []

    for level in range(1, 7):
        for h in soup.find_all(f'h{level}'):
            text = h.get_text(strip=True)[:100]
            headings.append({
                "level": level,
                "text": text,
                "tag": f"h{level}",
            })

    # SEO Checks
    h1_count = len([h for h in headings if h["level"] == 1])
    if h1_count == 0:
        issues.append({"severity": "warning", "message": "No H1 tag found — bad for SEO"})
    elif h1_count > 1:
        issues.append({"severity": "warning", "message": f"Multiple H1 tags found ({h1_count}) — use only one H1 per page"})

    # Check heading order
    prev_level = 0
    for h in headings:
        if h["level"] > prev_level + 1 and prev_level > 0:
            issues.append({
                "severity": "info",
                "message": f"Skipped heading level: H{prev_level} → H{h['level']} ('{h['text'][:30]}')"
            })
        prev_level = h["level"]

    if not issues:
        issues.append({"severity": "pass", "message": "Heading hierarchy looks good"})

    return {
        "headings": headings,
        "total_headings": len(headings),
        "heading_counts": {f"h{i}": len([h for h in headings if h["level"] == i]) for i in range(1, 7)},
        "issues": issues,
    }


# =============================================================================
# Link Analysis
# =============================================================================

def analyze_links(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Analyze internal and external links."""
    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc

    internal_links = []
    external_links = []
    anchor_links = []
    broken_refs = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)[:80]

        if href.startswith("#"):
            anchor_links.append({"href": href, "text": text})
        elif href.startswith("mailto:") or href.startswith("tel:") or href.startswith("javascript:"):
            continue
        else:
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            if parsed.netloc == base_domain:
                internal_links.append({"url": full_url, "text": text})
            else:
                external_links.append({"url": full_url, "text": text, "domain": parsed.netloc})

    # Find unique external domains
    ext_domains = Counter(l["domain"] for l in external_links)

    return {
        "total_links": len(internal_links) + len(external_links) + len(anchor_links),
        "internal_links": len(internal_links),
        "external_links": len(external_links),
        "anchor_links": len(anchor_links),
        "internal_urls": internal_links[:30],
        "external_domains": [{"domain": d, "count": c} for d, c in ext_domains.most_common(10)],
        "external_urls": external_links[:20],
    }


# =============================================================================
# Page Metadata
# =============================================================================

def analyze_metadata(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Extract and analyze page metadata."""
    meta = {}

    # Title
    title_tag = soup.find("title")
    meta["title"] = title_tag.get_text(strip=True) if title_tag else None
    meta["title_length"] = len(meta["title"]) if meta["title"] else 0

    # Meta description
    desc_tag = soup.find("meta", attrs={"name": "description"})
    meta["description"] = desc_tag["content"] if desc_tag and desc_tag.get("content") else None
    meta["description_length"] = len(meta["description"]) if meta["description"] else 0

    # Meta keywords
    kw_tag = soup.find("meta", attrs={"name": "keywords"})
    meta["keywords"] = kw_tag["content"] if kw_tag and kw_tag.get("content") else None

    # Canonical URL
    canonical = soup.find("link", attrs={"rel": "canonical"})
    meta["canonical_url"] = canonical["href"] if canonical and canonical.get("href") else None

    # Robots
    robots = soup.find("meta", attrs={"name": "robots"})
    meta["robots"] = robots["content"] if robots and robots.get("content") else None

    # OpenGraph
    og_tags = {}
    for og in soup.find_all("meta", attrs={"property": re.compile(r'^og:')}):
        prop = og.get("property", "").replace("og:", "")
        og_tags[prop] = og.get("content", "")[:200]
    meta["open_graph"] = og_tags

    # Twitter cards
    tw_tags = {}
    for tw in soup.find_all("meta", attrs={"name": re.compile(r'^twitter:')}):
        name = tw.get("name", "").replace("twitter:", "")
        tw_tags[name] = tw.get("content", "")[:200]
    meta["twitter_card"] = tw_tags

    # Favicon
    favicon = soup.find("link", attrs={"rel": re.compile(r'icon', re.I)})
    meta["favicon"] = urljoin(url, favicon["href"]) if favicon and favicon.get("href") else None

    # Language
    html_tag = soup.find("html")
    meta["language"] = html_tag.get("lang") if html_tag else None

    # Viewport
    viewport = soup.find("meta", attrs={"name": "viewport"})
    meta["viewport"] = viewport["content"] if viewport and viewport.get("content") else None

    # SEO Issues
    seo_issues = []
    if not meta["title"]:
        seo_issues.append({"severity": "critical", "message": "Missing page title"})
    elif meta["title_length"] > 60:
        seo_issues.append({"severity": "warning", "message": f"Title too long ({meta['title_length']} chars, recommended <60)"})

    if not meta["description"]:
        seo_issues.append({"severity": "warning", "message": "Missing meta description"})
    elif meta["description_length"] > 160:
        seo_issues.append({"severity": "info", "message": f"Description too long ({meta['description_length']} chars, recommended <160)"})

    if not meta["viewport"]:
        seo_issues.append({"severity": "warning", "message": "Missing viewport meta tag — site may not be mobile-friendly"})

    if not meta["canonical_url"]:
        seo_issues.append({"severity": "info", "message": "No canonical URL set"})

    if not og_tags:
        seo_issues.append({"severity": "info", "message": "No OpenGraph tags — social sharing may lack rich previews"})

    if not seo_issues:
        seo_issues.append({"severity": "pass", "message": "All metadata checks passed"})

    meta["seo_issues"] = seo_issues

    return meta


# =============================================================================
# Sitemap Builder
# =============================================================================

def build_sitemap_tree(internal_links: List[Dict[str, str]], base_url: str) -> Dict[str, Any]:
    """Build a hierarchical sitemap tree from internal links."""
    parsed_base = urlparse(base_url)
    tree = {"path": "/", "children": {}, "url": base_url}

    for link in internal_links:
        parsed = urlparse(link["url"])
        path = parsed.path.strip("/")
        parts = path.split("/") if path else []

        current = tree
        for part in parts:
            if part not in current["children"]:
                current["children"][part] = {
                    "path": part,
                    "children": {},
                    "url": link["url"],
                    "text": link.get("text", ""),
                }
            current = current["children"][part]

    def tree_to_list(node, depth=0):
        items = [{"path": node["path"], "depth": depth, "url": node.get("url", ""), "children_count": len(node["children"])}]
        for child in sorted(node["children"].values(), key=lambda x: x["path"]):
            items.extend(tree_to_list(child, depth + 1))
        return items

    flat_tree = tree_to_list(tree)

    return {
        "total_paths": len(flat_tree),
        "max_depth": max(n["depth"] for n in flat_tree) if flat_tree else 0,
        "tree": flat_tree[:50],  # Limit output
    }


# =============================================================================
# Resource Analysis
# =============================================================================

def analyze_resources(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Analyze page resources (scripts, styles, images)."""
    scripts = soup.find_all("script")
    stylesheets = soup.find_all("link", rel="stylesheet")
    images = soup.find_all("img")
    iframes = soup.find_all("iframe")

    external_scripts = [{"src": urljoin(url, s["src"])} for s in scripts if s.get("src")]
    inline_scripts = len([s for s in scripts if not s.get("src")])

    external_styles = [{"href": urljoin(url, s["href"])} for s in stylesheets if s.get("href")]
    inline_styles = len(soup.find_all("style"))

    img_list = []
    for img in images[:20]:
        src = img.get("src") or img.get("data-src")
        alt = img.get("alt", "")
        img_list.append({
            "src": urljoin(url, src) if src else None,
            "alt": alt[:60],
            "has_alt": bool(alt),
        })

    missing_alt = sum(1 for i in img_list if not i["has_alt"])

    return {
        "scripts": {
            "external": len(external_scripts),
            "inline": inline_scripts,
            "total": len(scripts),
            "sources": external_scripts[:10],
        },
        "stylesheets": {
            "external": len(external_styles),
            "inline": inline_styles,
            "total": len(stylesheets) + inline_styles,
            "sources": external_styles[:10],
        },
        "images": {
            "total": len(images),
            "missing_alt": missing_alt,
            "accessibility_score": f"{((len(images) - missing_alt) / max(len(images), 1)) * 100:.0f}%",
            "items": img_list[:10],
        },
        "iframes": len(iframes),
    }


# =============================================================================
# Main Analyzer
# =============================================================================

async def analyze_structure(url: str) -> Dict[str, Any]:
    """
    Run complete structure analysis on a URL.
    
    Returns comprehensive structure report.
    """
    import time
    start = time.time()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            response = await client.get(url, headers=headers)
            html = response.text
            status_code = response.status_code
    except Exception as e:
        return {
            "url": url,
            "error": f"Failed to fetch page: {str(e)}",
        }

    soup = BeautifulSoup(html, "html.parser")

    # Run all analyses
    dom = analyze_dom_depth(soup)
    heading_info = analyze_headings(soup)
    link_info = analyze_links(soup, url)
    meta_info = analyze_metadata(soup, url)
    sitemap = build_sitemap_tree(link_info["internal_urls"], url)
    resources = analyze_resources(soup, url)

    duration = time.time() - start

    # Compute overall health score
    issues_count = len([i for i in heading_info["issues"] if i["severity"] in ("warning", "critical")])
    issues_count += len([i for i in meta_info["seo_issues"] if i["severity"] in ("warning", "critical")])
    if resources["images"]["missing_alt"] > 5:
        issues_count += 1

    score = max(0, 100 - issues_count * 12)

    return {
        "url": url,
        "status_code": status_code,
        "score": score,
        "grade": "A" if score >= 90 else "B" if score >= 70 else "C" if score >= 50 else "D" if score >= 30 else "F",
        "dom": dom,
        "headings": heading_info,
        "links": link_info,
        "metadata": meta_info,
        "sitemap": sitemap,
        "resources": resources,
        "duration_seconds": round(duration, 2),
        "analyzed_at": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Test
# =============================================================================

async def test_analyzer():
    """Test the structure analyzer."""
    result = await analyze_structure("https://quotes.toscrape.com/")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(test_analyzer())
