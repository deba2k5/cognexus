"""
AWE Real-Time Web Extractor
============================
Uses Groq API to extract structured data from websites in real-time.
"""

import os
import asyncio
import httpx
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import json
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")


async def fetch_webpage(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Fetch a webpage and return its HTML content and metadata.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        
        return {
            "url": str(response.url),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "html": response.text,
        }


def extract_text_content(html: str, max_length: int = 15000) -> str:
    """
    Extract clean text content from HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    
    # Get text content
    text = soup.get_text(separator="\n", strip=True)
    
    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "\n...[truncated]"
    
    return text


def extract_links(html: str, base_url: str) -> List[Dict[str, str]]:
    """
    Extract all links from the page.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        
        # Make absolute URL
        if href.startswith("/"):
            from urllib.parse import urljoin
            href = urljoin(base_url, href)
        
        if href.startswith("http") and text:
            links.append({"url": href, "text": text[:100]})
    
    return links[:50]  # Limit to 50 links


async def extract_with_groq(
    text_content: str,
    url: str,
    objective: str,
    target_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Use Groq LLM to extract structured data from text content.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment")
    
    # Build the prompt
    fields_instruction = ""
    if target_fields:
        fields_instruction = f"\n\nTarget fields to extract: {', '.join(target_fields)}"
    
    prompt = f"""You are a data extraction expert. Analyze the following webpage content and extract structured data based on the objective.

URL: {url}
Objective: {objective}{fields_instruction}

WEBPAGE CONTENT:
{text_content}

INSTRUCTIONS:
1. Extract all relevant information that matches the objective
2. Return data as a JSON array of objects
3. Each object should represent one item (e.g., one person, one product, one article)
4. Include all relevant fields you can find
5. If specific fields were requested, prioritize those
6. Be thorough but accurate - only include information actually present in the content

Return ONLY valid JSON array, no other text. If no relevant data found, return empty array [].

Example format:
[
  {{"name": "...", "title": "...", "email": "..."}},
  {{"name": "...", "title": "...", "email": "..."}}
]

JSON OUTPUT:"""

    # Call Groq API
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
            }
        )
        response.raise_for_status()
        result = response.json()
    
    # Parse the response
    content = result["choices"][0]["message"]["content"]
    
    # Try to extract JSON from the response
    try:
        # Try direct parse
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                data = []
        else:
            data = []
    
    return {
        "data": data if isinstance(data, list) else [data],
        "model": MODEL_NAME,
        "tokens_used": result.get("usage", {}).get("total_tokens", 0),
    }


async def extract_from_url(
    url: str,
    objective: str = "Extract all relevant information",
    target_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main function to extract data from a URL using Groq.
    
    Args:
        url: The webpage URL to extract from
        objective: What to extract (e.g., "Extract faculty profiles")
        target_fields: Optional list of specific fields to extract
    
    Returns:
        Dictionary with extracted data and metadata
    """
    import time
    start_time = time.time()
    
    # Fetch the webpage
    page_data = await fetch_webpage(url)
    
    # Extract text content
    text_content = extract_text_content(page_data["html"])
    
    # Extract links for reference
    links = extract_links(page_data["html"], url)
    
    # Use Groq to extract structured data
    extraction = await extract_with_groq(
        text_content=text_content,
        url=url,
        objective=objective,
        target_fields=target_fields,
    )
    
    duration = time.time() - start_time
    
    return {
        "url": url,
        "objective": objective,
        "data": extraction["data"],
        "metadata": {
            "model": extraction["model"],
            "tokens_used": extraction["tokens_used"],
            "content_length": len(text_content),
            "links_found": len(links),
            "duration_seconds": round(duration, 2),
        },
        "links": links[:10],  # Include first 10 links for reference
    }


# Quick test function
async def test_extraction():
    """Test the extraction on a sample URL."""
    result = await extract_from_url(
        url="https://quotes.toscrape.com/",
        objective="Extract all quotes with their author names",
        target_fields=["quote", "author"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(test_extraction())
