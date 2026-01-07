"""
AWE Framework - Multi-Site Validation Test
===========================================

Tests the AWE framework against 5 different university faculty websites
to validate generalizability and robustness.

Test Sites:
1. WashU Olin Business School
2. MIT CSAIL Faculty
3. Stanford CS Faculty  
4. CMU School of Computer Science
5. UC Berkeley EECS Faculty
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from awe import WebExplorer, ExplorationGoal, AWEConfig


@dataclass
class TestSite:
    """A test site configuration."""
    name: str
    url: str
    expected_fields: List[str]
    min_expected_profiles: int
    max_test_profiles: int = 10  # Limit for quick testing


# =============================================================================
# Test Sites Configuration
# =============================================================================

TEST_SITES = [
    TestSite(
        name="Princeton CS Faculty",
        url="https://www.cs.princeton.edu/people/faculty",
        expected_fields=["name", "title", "email"],
        min_expected_profiles=30,
        max_test_profiles=5,
    ),
    TestSite(
        name="Cornell CS Faculty",
        url="https://www.cs.cornell.edu/people/faculty",
        expected_fields=["name", "title", "email"],
        min_expected_profiles=50,
        max_test_profiles=5,
    ),
    TestSite(
        name="UIUC CS Faculty",
        url="https://cs.illinois.edu/about/people/faculty",
        expected_fields=["name", "title", "email"],
        min_expected_profiles=100,
        max_test_profiles=5,
    ),
    TestSite(
        name="UW CS Faculty",
        url="https://www.cs.washington.edu/people/faculty",
        expected_fields=["name", "title", "email"],
        min_expected_profiles=50,
        max_test_profiles=5,
    ),
    TestSite(
        name="UCLA CS Faculty",
        url="https://www.cs.ucla.edu/people/faculty/",
        expected_fields=["name", "title", "email"],
        min_expected_profiles=40,
        max_test_profiles=5,
    ),
]


@dataclass
class TestResult:
    """Result of testing a single site."""
    site_name: str
    url: str
    success: bool
    profiles_found: int
    profiles_extracted: int
    extraction_rate: float
    fields_coverage: Dict[str, float]  # field -> % of profiles with this field
    duration_seconds: float
    errors: List[str]
    sample_profile: Optional[Dict[str, Any]] = None


# =============================================================================
# Test Runner
# =============================================================================

async def test_site(site: TestSite, config: AWEConfig) -> TestResult:
    """Test AWE on a single site."""
    print(f"\n{'='*70}")
    print(f"  Testing: {site.name}")
    print(f"  URL: {site.url}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    errors = []
    profiles_found = 0
    profiles_extracted = 0
    fields_coverage = {f: 0.0 for f in site.expected_fields}
    sample_profile = None
    
    try:
        goal = ExplorationGoal(
            objective=f"Extract faculty profile information from {site.name}",
            target_fields=site.expected_fields,
            start_url=site.url,
            constraints={
                "max_items": site.max_test_profiles,
                "domain_restrict": True,
            },
        )
        
        async with WebExplorer(config=config, verbose=True) as explorer:
            result = await explorer.explore(goal)
        
        profiles_found = result.pages_visited
        profiles_extracted = result.items_extracted
        
        # Calculate field coverage
        if result.data:
            for field in site.expected_fields:
                count = sum(1 for item in result.data 
                           if hasattr(item, 'data') and item.data.get(field)
                           or isinstance(item, dict) and item.get(field))
                fields_coverage[field] = count / len(result.data) if result.data else 0.0
            
            # Get sample profile
            if result.data:
                first = result.data[0]
                sample_profile = first.data if hasattr(first, 'data') else first
        
        errors = result.errors
        
    except Exception as e:
        errors.append(str(e))
        import traceback
        traceback.print_exc()
    
    duration = (datetime.now() - start_time).total_seconds()
    extraction_rate = profiles_extracted / profiles_found if profiles_found > 0 else 0.0
    
    return TestResult(
        site_name=site.name,
        url=site.url,
        success=profiles_extracted > 0 and len(errors) == 0,
        profiles_found=profiles_found,
        profiles_extracted=profiles_extracted,
        extraction_rate=extraction_rate,
        fields_coverage=fields_coverage,
        duration_seconds=duration,
        errors=errors,
        sample_profile=sample_profile,
    )


async def run_all_tests(sites: List[TestSite] = None) -> List[TestResult]:
    """Run tests on all sites."""
    sites = sites or TEST_SITES
    
    # Configuration for testing (faster settings)
    config = AWEConfig(
        model="gemma3:12b",
        temperature=0.3,
        max_tokens=2048,
        
        # ToT settings (reduced for speed)
        max_thoughts=3,
        max_depth=2,
        search_strategy="bfs",
        
        # Browser settings
        headless=True,  # Run headless for testing
        timeout=30000,
        wait_after_navigation=2000,
        request_delay=1.0,
        viewport=(1920, 1080),
        
        # Learning settings
        learning_enabled=True,
        min_examples_for_pattern=2,
        save_templates=True,
        
        # Reliability
        max_retries=2,
        retry_delay=1.0,
    )
    
    results = []
    
    for site in sites:
        result = await test_site(site, config)
        results.append(result)
        
        # Print quick result
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"\n{status} - {site.name}: {result.profiles_extracted} profiles extracted\n")
    
    return results


def print_summary(results: List[TestResult]):
    """Print a summary of all test results."""
    print("\n" + "="*70)
    print("  AWE FRAMEWORK - MULTI-SITE VALIDATION SUMMARY")
    print("="*70)
    
    total_sites = len(results)
    passed = sum(1 for r in results if r.success)
    total_profiles = sum(r.profiles_extracted for r in results)
    total_time = sum(r.duration_seconds for r in results)
    
    print(f"\n  Sites Tested: {total_sites}")
    print(f"  Sites Passed: {passed}/{total_sites} ({passed/total_sites*100:.0f}%)")
    print(f"  Total Profiles: {total_profiles}")
    print(f"  Total Time: {total_time:.1f}s")
    
    print("\n" + "-"*70)
    print("  DETAILED RESULTS")
    print("-"*70)
    
    for r in results:
        status = "✅" if r.success else "❌"
        print(f"\n  {status} {r.site_name}")
        print(f"     URL: {r.url}")
        print(f"     Profiles: {r.profiles_extracted} extracted / {r.profiles_found} found")
        print(f"     Extraction Rate: {r.extraction_rate*100:.0f}%")
        print(f"     Duration: {r.duration_seconds:.1f}s")
        
        if r.fields_coverage:
            coverage_str = ", ".join(f"{k}: {v*100:.0f}%" for k, v in r.fields_coverage.items())
            print(f"     Field Coverage: {coverage_str}")
        
        if r.errors:
            print(f"     Errors: {len(r.errors)}")
            for err in r.errors[:3]:
                print(f"       - {err[:80]}")
        
        if r.sample_profile:
            print(f"     Sample: {json.dumps(r.sample_profile, indent=2)[:200]}...")
    
    print("\n" + "="*70)
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "validation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = []
    for r in results:
        results_data.append({
            "site_name": r.site_name,
            "url": r.url,
            "success": r.success,
            "profiles_found": r.profiles_found,
            "profiles_extracted": r.profiles_extracted,
            "extraction_rate": r.extraction_rate,
            "fields_coverage": r.fields_coverage,
            "duration_seconds": r.duration_seconds,
            "errors": r.errors,
            "sample_profile": r.sample_profile,
        })
    
    with open(output_path / f"validation_{timestamp}.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_sites": total_sites,
            "passed": passed,
            "total_profiles": total_profiles,
            "total_time": total_time,
            "results": results_data,
        }, f, indent=2)
    
    print(f"\n  Results saved to: {output_path / f'validation_{timestamp}.json'}")
    
    return passed == total_sites


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("  AWE FRAMEWORK - MULTI-SITE VALIDATION TEST")
    print("  Testing against 5 different university faculty websites")
    print("="*70)
    
    results = await run_all_tests()
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
