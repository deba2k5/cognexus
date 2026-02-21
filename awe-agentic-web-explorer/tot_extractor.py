"""
AWE Tree of Thought Extractor
==============================
Implements ToT-based extraction for improved accuracy with SLMs.

Uses multi-path exploration:
1. Generate multiple extraction strategies
2. Evaluate and score each approach
3. Execute best strategy with fallback
4. Self-reflect and improve
"""

import os
import asyncio
import httpx
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
TOT_ENABLED = os.getenv("TOT_ENABLED", "true").lower() == "true"
TOT_MAX_THOUGHTS = int(os.getenv("TOT_MAX_THOUGHTS", "3"))
TOT_SEARCH_STRATEGY = os.getenv("TOT_SEARCH_STRATEGY", "beam")


# =============================================================================
# Data Classes
# =============================================================================

class ThoughtStatus(str, Enum):
    PENDING = "pending"
    EXPLORING = "exploring"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class ExtractionThought:
    """A single extraction strategy/approach."""
    id: str
    strategy: str  # Description of the extraction approach
    reasoning: str  # Why this might work
    selectors: List[str] = field(default_factory=list)  # CSS selectors to try
    fields: List[str] = field(default_factory=list)  # Expected fields
    
    feasibility: float = 0.5
    confidence: float = 0.5
    value: float = 0.5
    
    status: ThoughtStatus = ThoughtStatus.PENDING
    result: Optional[List[Dict]] = None
    error: Optional[str] = None
    
    @property
    def score(self) -> float:
        """Combined score for ranking."""
        return (self.feasibility + self.confidence + self.value) / 3


# =============================================================================
# Groq API Client
# =============================================================================

async def call_groq(prompt: str, temperature: float = 0.3, max_tokens: int = 4096, max_retries: int = 3) -> str:
    """Call Groq API for LLM completion with retry on rate limits."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    
    logger.info(f"Calling Groq API with model={MODEL_NAME}, prompt_len={len(prompt)}, temp={temperature}")
    
    for attempt in range(max_retries):
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            
            if response.status_code == 429:
                # Rate limited — wait and retry with exponential backoff
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                await asyncio.sleep(wait_time)
                continue
            
            if response.status_code != 200:
                logger.error(f"Groq API error {response.status_code}: {response.text[:500]}")
                response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
    
    raise Exception(f"Groq API rate limited after {max_retries} retries")


# =============================================================================
# Thought Generator
# =============================================================================

STRATEGY_GENERATION_PROMPT = """You are an expert web scraper. Given this webpage content, generate {n} different extraction strategies.

URL: {url}
OBJECTIVE: {objective}

CONTENT PREVIEW (first 3000 chars):
{content}

Generate {n} distinct approaches to extract structured data. For each approach:
1. Describe the extraction strategy
2. Explain why it might work for this content
3. Suggest CSS-like selectors if you can identify patterns
4. List the fields you expect to extract
5. Rate feasibility (0-1), confidence (0-1), and value (0-1)

Output as JSON array:
[
  {{
    "strategy": "Look for repeated patterns like cards or list items",
    "reasoning": "The page has repeating div.item elements containing quotes",
    "selectors": [".item", ".quote-text", ".author"],
    "fields": ["quote", "author", "tags"],
    "feasibility": 0.8,
    "confidence": 0.7,
    "value": 0.9
  }}
]

Focus on DIFFERENT approaches - not just variations of the same idea.
"""


async def generate_extraction_thoughts(
    content: str,
    url: str,
    objective: str,
    n: int = 3,
) -> List[ExtractionThought]:
    """Generate multiple extraction strategies using LLM."""
    import uuid
    
    prompt = STRATEGY_GENERATION_PROMPT.format(
        n=n,
        url=url,
        objective=objective,
        content=content[:3000],
    )
    
    try:
        response = await call_groq(prompt, temperature=0.5)
        
        # Parse JSON response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            logger.warning("No JSON found in strategy generation")
            return []
        
        data = json.loads(json_match.group())
        
        thoughts = []
        for item in data:
            if isinstance(item, dict):
                thoughts.append(ExtractionThought(
                    id=str(uuid.uuid4()),
                    strategy=item.get("strategy", "Default extraction"),
                    reasoning=item.get("reasoning", ""),
                    selectors=item.get("selectors", []),
                    fields=item.get("fields", []),
                    feasibility=float(item.get("feasibility", 0.5)),
                    confidence=float(item.get("confidence", 0.5)),
                    value=float(item.get("value", 0.5)),
                ))
        
        logger.info(f"Generated {len(thoughts)} extraction strategies")
        return thoughts
        
    except Exception as e:
        logger.error(f"Strategy generation failed: {e}")
        # Return default thought
        return [ExtractionThought(
            id="default",
            strategy="Direct LLM extraction without specific strategy",
            reasoning="Fallback when strategy generation fails",
            feasibility=0.5,
            confidence=0.5,
            value=0.5,
        )]


# =============================================================================
# Thought Evaluator
# =============================================================================

EVALUATION_PROMPT = """Evaluate this extraction strategy for the given content.

STRATEGY: {strategy}
REASONING: {reasoning}

CONTENT STRUCTURE:
- Content length: {content_length} chars
- Has tables: {has_tables}
- Has lists: {has_lists}
- Has cards/items: {has_cards}

Should we try this strategy? Rate:
- adjusted_feasibility (0-1): Can this actually be done?
- adjusted_confidence (0-1): Will it produce good results?
- adjusted_value (0-1): Is this the best approach?

Output JSON:
{{
  "should_try": true,
  "adjusted_feasibility": 0.8,
  "adjusted_confidence": 0.7,
  "adjusted_value": 0.9,
  "reason": "Brief explanation"
}}
"""


async def evaluate_thought(
    thought: ExtractionThought,
    content: str,
) -> ExtractionThought:
    """Evaluate and adjust a thought's scores."""
    # Quick heuristic checks
    has_tables = "<table" in content.lower()
    has_lists = "<li" in content.lower() or "<ul" in content.lower()
    has_cards = any(p in content.lower() for p in ["class=\"card", "class=\"item", "class=\"quote"])
    
    prompt = EVALUATION_PROMPT.format(
        strategy=thought.strategy,
        reasoning=thought.reasoning,
        content_length=len(content),
        has_tables=has_tables,
        has_lists=has_lists,
        has_cards=has_cards,
    )
    
    try:
        response = await call_groq(prompt, temperature=0.1, max_tokens=500)
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            thought.feasibility = float(data.get("adjusted_feasibility", thought.feasibility))
            thought.confidence = float(data.get("adjusted_confidence", thought.confidence))
            thought.value = float(data.get("adjusted_value", thought.value))
            
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
    
    return thought


async def evaluate_thoughts(
    thoughts: List[ExtractionThought],
    content: str,
) -> List[ExtractionThought]:
    """Evaluate thoughts sequentially to avoid rate limits."""
    evaluated = []
    for t in thoughts:
        result = await evaluate_thought(t, content)
        evaluated.append(result)
        # Small delay between evaluations to avoid rate limits
        await asyncio.sleep(1)
    return evaluated


# =============================================================================
# Thought Executor
# =============================================================================

EXTRACTION_PROMPT = """Extract structured data from this content using the following strategy.

STRATEGY: {strategy}
EXPECTED FIELDS: {fields}

CONTENT:
{content}

Extract ALL relevant items. Return as JSON array:
[
  {{"field1": "value1", "field2": "value2", ...}},
  ...
]

Be thorough - extract every item that matches. Return ONLY valid JSON."""


async def execute_thought(
    thought: ExtractionThought,
    content: str,
    url: str,
) -> ExtractionThought:
    """Execute an extraction strategy."""
    thought.status = ThoughtStatus.EXPLORING
    
    prompt = EXTRACTION_PROMPT.format(
        strategy=thought.strategy,
        fields=", ".join(thought.fields) if thought.fields else "any relevant fields",
        content=content[:12000],  # Limit content size
    )
    
    try:
        response = await call_groq(prompt, temperature=0.1)
        
        # Parse JSON — handle extra text around JSON arrays
        json_match = re.search(r'\[\s*\{', response)
        if json_match:
            # Find the start of the JSON array
            json_start = json_match.start()
            # Try to find the matching end bracket
            bracket_count = 0
            json_end = json_start
            for i in range(json_start, len(response)):
                if response[i] == '[':
                    bracket_count += 1
                elif response[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_end = i + 1
                        break
            
            try:
                data = json.loads(response[json_start:json_end])
            except json.JSONDecodeError:
                # Fallback: try regex approach
                fallback_match = re.search(r'\[[\s\S]*?\](?=\s*$|\s*```)', response)
                data = json.loads(fallback_match.group()) if fallback_match else []
            
            if isinstance(data, list) and len(data) > 0:
                thought.result = data
                thought.status = ThoughtStatus.SUCCEEDED
                logger.info(f"Strategy '{thought.strategy[:30]}...' extracted {len(data)} items")
            else:
                thought.status = ThoughtStatus.FAILED
                thought.error = "Empty extraction result"
        else:
            thought.status = ThoughtStatus.FAILED
            thought.error = "No JSON in response"
            
    except Exception as e:
        thought.status = ThoughtStatus.FAILED
        thought.error = str(e)
        logger.error(f"Execution failed: {e}")
    
    return thought


# =============================================================================
# ToT Engine
# =============================================================================

async def extract_with_tot(
    content: str,
    url: str,
    objective: str,
    max_thoughts: int = None,
) -> Dict[str, Any]:
    """
    Main ToT extraction function.
    
    1. Generate multiple extraction strategies
    2. Evaluate each strategy
    3. Execute best strategies in order
    4. Return first successful result
    """
    import time
    start_time = time.time()
    
    max_thoughts = max_thoughts or TOT_MAX_THOUGHTS
    
    result = {
        "data": [],
        "thoughts_generated": 0,
        "thoughts_evaluated": 0,
        "thoughts_tried": 0,
        "best_strategy": None,
        "all_strategies": [],
        "tot_enabled": True,
    }
    
    # Step 1: Generate thoughts
    logger.info(f"ToT: Generating {max_thoughts} extraction strategies...")
    thoughts = await generate_extraction_thoughts(content, url, objective, n=max_thoughts)
    result["thoughts_generated"] = len(thoughts)
    
    if not thoughts:
        return result
    
    # Step 2: Evaluate thoughts
    logger.info("ToT: Evaluating strategies...")
    thoughts = await evaluate_thoughts(thoughts, content)
    result["thoughts_evaluated"] = len(thoughts)
    
    # Sort by score
    thoughts.sort(key=lambda t: t.score, reverse=True)
    
    # Record all strategies for transparency
    result["all_strategies"] = [
        {
            "strategy": t.strategy,
            "reasoning": t.reasoning,
            "score": round(t.score, 2),
            "status": t.status.value,
        }
        for t in thoughts
    ]
    
    # Step 3: Execute thoughts in order of score
    for thought in thoughts:
        if thought.score < 0.3:  # Skip low-scoring thoughts
            thought.status = ThoughtStatus.ABANDONED
            continue
        
        logger.info(f"ToT: Trying strategy (score={thought.score:.2f}): {thought.strategy[:50]}...")
        result["thoughts_tried"] += 1
        
        thought = await execute_thought(thought, content, url)
        
        if thought.status == ThoughtStatus.SUCCEEDED and thought.result:
            result["data"] = thought.result
            result["best_strategy"] = {
                "strategy": thought.strategy,
                "reasoning": thought.reasoning,
                "score": round(thought.score, 2),
                "items_extracted": len(thought.result),
            }
            break
    
    result["duration_seconds"] = round(time.time() - start_time, 2)
    
    logger.info(f"ToT: Complete. Generated={result['thoughts_generated']}, "
                f"Tried={result['thoughts_tried']}, "
                f"Extracted={len(result['data'])} items")
    
    return result


# =============================================================================
# Main Extraction Function
# =============================================================================

async def extract_from_url_with_tot(
    url: str,
    objective: str = "Extract all relevant structured information",
    use_tot: bool = None,
) -> Dict[str, Any]:
    """
    Extract data from URL using ToT if enabled.
    
    Args:
        url: Website URL to extract from
        objective: What to extract
        use_tot: Force ToT on/off (uses env config if None)
    
    Returns:
        Extraction result with data and metadata
    """
    from extractor import fetch_webpage, extract_text_content
    
    # Fetch page
    page_data = await fetch_webpage(url)
    text_content = extract_text_content(page_data["html"])
    
    # Determine if ToT should be used
    if use_tot is None:
        use_tot = TOT_ENABLED
    
    if use_tot:
        # Use ToT extraction
        result = await extract_with_tot(
            content=text_content,
            url=url,
            objective=objective,
        )
        
        return {
            "url": url,
            "objective": objective,
            "data": result["data"],
            "metadata": {
                "model": MODEL_NAME,
                "tot_enabled": True,
                "thoughts_generated": result["thoughts_generated"],
                "thoughts_tried": result["thoughts_tried"],
                "best_strategy": result.get("best_strategy"),
                "all_strategies": result.get("all_strategies", []),
                "duration_seconds": result.get("duration_seconds", 0),
            },
        }
    else:
        # Fall back to simple extraction
        from extractor import extract_from_url
        return await extract_from_url(url, objective)


# =============================================================================
# Test
# =============================================================================

async def test_tot_extraction():
    """Test ToT extraction."""
    result = await extract_from_url_with_tot(
        url="https://quotes.toscrape.com/",
        objective="Extract all quotes with authors and tags",
        use_tot=True,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_tot_extraction())
