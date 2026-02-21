"""
WashU Olin Faculty Crawler Example
===================================

Demonstrates AWE framework on the original use case:
extracting faculty profiles from WashU Olin Business School.

This example shows:
1. Goal definition
2. Custom configuration
3. Progress tracking
4. Result export
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load env from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from awe import (
    WebExplorer,
    ExplorationGoal,
    AWEConfig,
    ExplorationResult,
)


# =============================================================================
# Configuration
# =============================================================================

WASHU_FACULTY_URL = "https://olin.wustl.edu/faculty-and-research/faculty/"

TARGET_FIELDS = [
    "name",
    "title",
    "email",
    "phone",
    "office",
    "department",
    "research_interests",
    "education",
    "bio",
    "profile_url",
]

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "washu_olin"


# =============================================================================
# Progress Display
# =============================================================================

class ProgressDisplay:
    """Visual progress display for exploration."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.phase = ""
        self.current = 0
        self.total = 0
    
    def update(self, data: dict):
        """Update progress display."""
        self.phase = data.get("phase", self.phase)
        self.current = data.get("current", self.current)
        self.total = data.get("total", self.total)
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.total > 0:
            pct = self.current / self.total * 100
            bar_len = 30
            filled = int(bar_len * self.current / self.total)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"\r  Progress: [{bar}] {pct:.0f}% ({self.current}/{self.total}) | {elapsed:.0f}s", end="")


# =============================================================================
# Result Export
# =============================================================================

def export_results(result: ExplorationResult, output_dir: Path):
    """Export extraction results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export summary
    summary = {
        "timestamp": timestamp,
        "url": WASHU_FACULTY_URL,
        "items_extracted": result.items_extracted,
        "pages_visited": result.pages_visited,
        "duration_seconds": result.duration_seconds,
        "errors": result.errors,
        "patterns_learned": len(result.patterns),
    }
    
    with open(output_dir / f"summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Export all data as one JSON
    all_data = []
    for item in result.data:
        if hasattr(item, "data"):
            all_data.append(item.data)
        elif isinstance(item, dict):
            all_data.append(item)
    
    with open(output_dir / f"faculty_{timestamp}.json", "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    # Export individual profiles
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    
    for item in all_data:
        name = item.get("name", "unknown").replace(" ", "_").replace("/", "_")
        filename = profiles_dir / f"{name}.json"
        
        with open(filename, "w") as f:
            json.dump(item, f, indent=2, ensure_ascii=False)
    
    # Export patterns
    if result.patterns:
        patterns_data = []
        for p in result.patterns:
            patterns_data.append({
                "domain": p.domain,
                "page_type": p.page_type.value if hasattr(p.page_type, "value") else str(p.page_type),
                "selectors": p.selectors,
                "usage_count": p.usage_count,
                "success_rate": p.success_rate,
            })
        
        with open(output_dir / f"patterns_{timestamp}.json", "w") as f:
            json.dump(patterns_data, f, indent=2)
    
    print(f"\nResults exported to: {output_dir}")
    print(f"  - Summary: summary_{timestamp}.json")
    print(f"  - Faculty data: faculty_{timestamp}.json")
    print(f"  - Individual profiles: profiles/")
    if result.patterns:
        print(f"  - Patterns: patterns_{timestamp}.json")
    
    return output_dir


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main entry point."""
    print("=" * 70)
    print("  WashU Olin Business School - Faculty Crawler")
    print("  Powered by AWE (Agentic Web Explorer)")
    print("=" * 70)
    print()
    
    # Define the exploration goal
    goal = ExplorationGoal(
        objective=(
            "Extract comprehensive faculty profile information from the "
            "WashU Olin Business School faculty directory. For each faculty "
            "member, extract their name, title, contact information, research "
            "interests, educational background, and biography."
        ),
        target_fields=TARGET_FIELDS,
        start_url=WASHU_FACULTY_URL,
        constraints={
            "max_items": 200,  # Limit for testing
            "domain_restrict": True,  # Stay on olin.wustl.edu
        },
    )
    
    # Custom configuration for WashU site
    config = AWEConfig(
        # LLM settings
        model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_provider=os.getenv("MODEL_PROVIDER", "groq"),
        temperature=0.3,
        max_tokens=4096,
        
        # ToT settings
        tot_max_thoughts=5,
        tot_max_depth=3,
        tot_search_strategy="beam",  # More balanced exploration
        
        # Browser settings
        headless=False,  # Set True for production
        timeout=30000,
        wait_after_navigation=2000,
        request_delay=1.0,
        viewport=(1920, 1080),
        
        # Learning settings
        learning_enabled=True,
        # min_examples_for_pattern=3, # Removed as it is not in AWEConfig
        save_templates=True,
        
        # Reliability settings
        max_retries=3,
        retry_backoff=2.0,
    )
    
    # Progress display
    progress = ProgressDisplay()
    
    try:
        # Run exploration
        async with WebExplorer(config=config, verbose=True) as explorer:
            result = await explorer.explore(
                goal=goal,
                progress_callback=progress.update,
            )
        
        print("\n")
        
        # Export results
        export_results(result, OUTPUT_DIR)
        
        # Print summary stats
        print("\n" + "=" * 70)
        print("  Extraction Summary")
        print("=" * 70)
        print(f"  Faculty profiles extracted: {result.items_extracted}")
        print(f"  Total pages visited: {result.pages_visited}")
        print(f"  Time taken: {result.duration_seconds:.1f} seconds")
        print(f"  Patterns learned: {len(result.patterns)}")
        
        if result.errors:
            print(f"\n  Errors encountered: {len(result.errors)}")
            for err in result.errors[:5]:  # Show first 5
                print(f"    - {err}")
        
        print()
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(main())
