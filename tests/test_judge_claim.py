"""
Tests for pipeline.judge_claim module.
"""

import pytest

from pipeline.judge_claim import (
    add_disclaimer_if_needed,
    judge_rule_based,
)
from core.constants import HEALTH_LEGAL_DISCLAIMER, VerdictEnum
from core.models import Claim, Evidence, Verdict


class TestJudgeRuleBased:
    """Tests for judge_rule_based function."""
    
    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim."""
        return Claim(
            id=1,
            raw_item_id=1,
            claim_text="Drinking 8 glasses of water daily is necessary.",
            topic="health",
            language="en",
            score=80,
            status="has_evidence",
        )
    
    def test_no_evidence_returns_unverified(self, sample_claim):
        """Test that no evidence returns Unverified verdict."""
        verdict = judge_rule_based(sample_claim, [])
        
        assert verdict.verdict == VerdictEnum.UNVERIFIED.value
        assert verdict.confidence < 0.5
    
    def test_with_factcheck_evidence(self, sample_claim):
        """Test with factcheck evidence."""
        evidence = [
            Evidence(
                id=1,
                claim_id=1,
                query="water",
                source_name="Snopes",
                source_type="factcheck",
                url="https://snopes.com/test",
                title="Water Myth",
                snippet="This claim is false.",
                credibility_score=80,
            )
        ]
        
        verdict = judge_rule_based(sample_claim, evidence)
        
        assert verdict.verdict == VerdictEnum.FALSE.value
        assert verdict.confidence >= 0.6
    
    def test_verdict_has_explanation(self, sample_claim):
        """Test that verdict has proper explanation."""
        evidence = [
            Evidence(
                id=1,
                claim_id=1,
                query="test",
                source_name="Wikipedia",
                source_type="wikipedia",
                url="https://wikipedia.org/test",
                title="Test Article",
                snippet="Information about the topic.",
                credibility_score=70,
            )
        ]
        
        verdict = judge_rule_based(sample_claim, evidence)
        
        assert "one_line_verdict" in verdict.explanation_json
        assert "why_believed" in verdict.explanation_json
        assert "what_wrong" in verdict.explanation_json
        assert "truth" in verdict.explanation_json


class TestAddDisclaimer:
    """Tests for add_disclaimer_if_needed function."""
    
    def test_health_topic_adds_disclaimer(self):
        """Test that health topic adds disclaimer."""
        verdict = Verdict(
            claim_id=1,
            verdict="False",
            explanation_json={"one_line_verdict": "test"},
            confidence=0.8,
        )
        
        result = add_disclaimer_if_needed(verdict, "health")
        
        assert "disclaimer" in result.explanation_json
        assert result.explanation_json["disclaimer"] == HEALTH_LEGAL_DISCLAIMER
    
    def test_non_health_topic_no_disclaimer(self):
        """Test that non-health topics don't add disclaimer."""
        verdict = Verdict(
            claim_id=1,
            verdict="False",
            explanation_json={"one_line_verdict": "test"},
            confidence=0.8,
        )
        
        result = add_disclaimer_if_needed(verdict, "history")
        
        assert result.explanation_json.get("disclaimer") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
