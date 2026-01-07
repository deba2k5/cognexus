"""
AWE Validator Agent
===================
Validates extraction results and suggests corrections.

The Validator:
1. Checks extracted data for quality
2. Identifies missing or incorrect fields
3. Suggests re-extraction strategies
4. Maintains quality metrics
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .base import BaseAgent, AgentCapability
from ..core.types import (
    AgentContext,
    AgentRole,
    ExtractionResult,
)
from ..core.config import AWEConfig


logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report from validating an extraction."""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    field_scores: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class ValidatorAgent(BaseAgent):
    """
    Quality assurance agent.
    
    Responsibilities:
    - Validate extracted data
    - Check data quality and completeness
    - Identify patterns in failures
    - Suggest corrections
    """
    
    # Validation rules for each field type
    FIELD_RULES = {
        "name": {
            "min_length": 3,
            "max_length": 100,
            "pattern": r'^[A-Z][a-zA-Z\-\'\.\s]+$',
            "required": True,
        },
        "email": {
            "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "required": False,
        },
        "phone": {
            "pattern": r'[\d\-\(\)\s\+\.]{7,}',
            "required": False,
        },
        "title": {
            "min_length": 5,
            "max_length": 200,
            "required": False,
        },
        "department": {
            "min_length": 3,
            "required": False,
        },
        "bio": {
            "min_length": 20,
            "max_length": 5000,
            "required": False,
        },
    }
    
    def __init__(
        self,
        config: AWEConfig,
        llm_func: Optional[Callable] = None,
        state=None,
    ):
        super().__init__(AgentRole.VALIDATOR, config, llm_func, state)
        
        # Track validation statistics
        self._stats = {
            "total_validated": 0,
            "passed": 0,
            "failed": 0,
            "field_failures": {},
        }
        
        self.register_capability(AgentCapability(
            name="validation",
            description="Validate extraction quality and suggest corrections",
            required_inputs=["extraction_result"],
            outputs=["validation_report"],
        ))
    
    async def process(self, context: AgentContext) -> Dict[str, Any]:
        """
        Validate extractions from context.
        """
        results = context.extracted_items
        if not results:
            return {"error": "No extraction results to validate"}
        
        reports = [self.validate(r) for r in results]
        
        return {
            "reports": reports,
            "pass_rate": sum(1 for r in reports if r.is_valid) / len(reports),
            "avg_quality": sum(r.quality_score for r in reports) / len(reports),
        }
    
    def validate(self, result: ExtractionResult) -> ValidationReport:
        """
        Validate an extraction result.
        
        Args:
            result: ExtractionResult to validate
        
        Returns:
            ValidationReport with detailed feedback
        """
        self._stats["total_validated"] += 1
        
        errors = []
        warnings = []
        suggestions = []
        field_scores = {}
        
        data = result.data
        
        # Validate each field
        for field_name, rules in self.FIELD_RULES.items():
            value = data.get(field_name)
            score, field_errors, field_warnings = self._validate_field(
                field_name, value, rules
            )
            
            field_scores[field_name] = score
            errors.extend(field_errors)
            warnings.extend(field_warnings)
            
            if score < 0.5:
                self._stats["field_failures"][field_name] = \
                    self._stats["field_failures"].get(field_name, 0) + 1
        
        # Check for cross-field issues
        cross_errors = self._cross_validate(data)
        errors.extend(cross_errors)
        
        # Generate suggestions
        if errors or warnings:
            suggestions = self._generate_suggestions(data, errors, warnings)
        
        # Calculate overall quality
        quality_score = self._calculate_quality(data, field_scores, errors)
        is_valid = len(errors) == 0 and quality_score >= 0.5
        
        if is_valid:
            self._stats["passed"] += 1
        else:
            self._stats["failed"] += 1
        
        return ValidationReport(
            is_valid=is_valid,
            quality_score=quality_score,
            field_scores=field_scores,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )
    
    def _validate_field(
        self,
        field_name: str,
        value: Any,
        rules: Dict,
    ) -> tuple[float, List[str], List[str]]:
        """
        Validate a single field.
        
        Returns (score, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check required
        if rules.get("required") and not value:
            errors.append(f"Missing required field: {field_name}")
            return 0.0, errors, warnings
        
        if not value:
            return 0.5, errors, warnings  # Optional field, no penalty
        
        # Convert to string for validation
        str_value = str(value) if not isinstance(value, str) else value
        
        # Check length
        min_len = rules.get("min_length", 0)
        max_len = rules.get("max_length", float("inf"))
        
        if len(str_value) < min_len:
            errors.append(f"{field_name} too short: {len(str_value)} < {min_len}")
            return 0.3, errors, warnings
        
        if len(str_value) > max_len:
            warnings.append(f"{field_name} too long: {len(str_value)} > {max_len}")
        
        # Check pattern
        pattern = rules.get("pattern")
        if pattern and not re.match(pattern, str_value):
            if field_name == "name":
                # Name pattern failure is serious
                errors.append(f"{field_name} doesn't match expected pattern")
                return 0.2, errors, warnings
            else:
                warnings.append(f"{field_name} format may be incorrect")
        
        return 1.0, errors, warnings
    
    def _cross_validate(self, data: Dict[str, Any]) -> List[str]:
        """Check for cross-field validation issues."""
        errors = []
        
        # Name shouldn't appear in title
        name = data.get("name", "")
        title = data.get("title", "")
        if name and title and name.lower() == title.lower():
            errors.append("Title appears to be the same as name")
        
        # Bio shouldn't be too similar to name/title
        bio = data.get("bio", "")
        if bio and name and len(bio) < 50 and name.lower() in bio.lower():
            errors.append("Bio is too short or just contains name")
        
        # Email domain should match expected patterns
        email = data.get("email", "")
        if email:
            domain = email.split("@")[-1] if "@" in email else ""
            if domain and not any(edu in domain for edu in ['.edu', '.ac.', 'university', 'college']):
                # Not necessarily an error, just unusual
                pass
        
        return errors
    
    def _calculate_quality(
        self,
        data: Dict[str, Any],
        field_scores: Dict[str, float],
        errors: List[str],
    ) -> float:
        """Calculate overall quality score."""
        if not field_scores:
            return 0.0
        
        # Base score from field scores
        avg_field_score = sum(field_scores.values()) / len(field_scores)
        
        # Penalty for errors
        error_penalty = min(0.3, len(errors) * 0.1)
        
        # Bonus for completeness
        non_empty = sum(1 for v in data.values() if v)
        completeness_bonus = (non_empty / max(len(self.FIELD_RULES), 1)) * 0.2
        
        quality = avg_field_score - error_penalty + completeness_bonus
        return max(0.0, min(1.0, quality))
    
    def _generate_suggestions(
        self,
        data: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> List[str]:
        """Generate suggestions for fixing issues."""
        suggestions = []
        
        # Check for common issues
        if "Missing required field: name" in errors:
            suggestions.append("Try using 'h1' or '.faculty-name' selector for name")
        
        if any("email" in e.lower() for e in errors + warnings):
            suggestions.append("Look for 'a[href^=\"mailto:\"]' for email")
        
        if any("too short" in e.lower() for e in errors):
            suggestions.append("The page structure may be different - try LLM extraction")
        
        # Suggest LLM fallback if many issues
        if len(errors) >= 3:
            suggestions.append("Consider using LLM-based extraction for this page type")
        
        return suggestions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            **self._stats,
            "pass_rate": self._stats["passed"] / max(self._stats["total_validated"], 1),
            "most_failed_fields": sorted(
                self._stats["field_failures"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }
    
    def validate_batch(self, results: List[ExtractionResult]) -> Dict[str, Any]:
        """
        Validate a batch of results.
        
        Returns aggregate statistics.
        """
        reports = [self.validate(r) for r in results]
        
        return {
            "total": len(reports),
            "passed": sum(1 for r in reports if r.is_valid),
            "failed": sum(1 for r in reports if not r.is_valid),
            "avg_quality": sum(r.quality_score for r in reports) / len(reports) if reports else 0,
            "common_errors": self._find_common_errors(reports),
            "reports": reports,
        }
    
    def _find_common_errors(self, reports: List[ValidationReport]) -> List[str]:
        """Find the most common errors across reports."""
        error_counts = {}
        
        for report in reports:
            for error in report.errors:
                # Normalize error message
                key = re.sub(r'\d+', 'N', error)  # Replace numbers
                error_counts[key] = error_counts.get(key, 0) + 1
        
        # Return top 5
        return sorted(error_counts.keys(), key=lambda k: error_counts[k], reverse=True)[:5]
