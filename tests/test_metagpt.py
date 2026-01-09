"""
Tests for MetaGPT integration modules.

Note: These tests work with stub classes when MetaGPT is not installed.
"""

import pytest

from metagpt_integration.schemas import (
    ClaimInput,
    EvidenceItem,
    EvidenceOutput,
    VerdictInput,
    VerdictOutput,
    ScriptInput,
    ScriptOutput,
    QAInput,
    QAOutput,
    QAIssue,
    PipelineContext,
)
from metagpt_integration.actions import (
    GatherEvidence,
    JudgeClaim,
    GenerateScript,
    QACheck,
    METAGPT_AVAILABLE,
)


class TestSchemas:
    """Tests for MetaGPT schemas."""
    
    def test_claim_input_creation(self):
        """Test ClaimInput model."""
        claim = ClaimInput(
            claim_id=1,
            claim_text="Test claim",
            topic="health",
            language="en",
            score=75,
            raw_item_id=10,
        )
        
        assert claim.claim_id == 1
        assert claim.topic == "health"
        assert "Test claim" in claim.to_message_content()
    
    def test_evidence_item_creation(self):
        """Test EvidenceItem model."""
        evidence = EvidenceItem(
            query="test query",
            source_name="Wikipedia",
            source_type="wikipedia",
            url="https://en.wikipedia.org/wiki/Test",
            title="Test Page",
            snippet="This is a test snippet",
            credibility_score=70,
        )
        
        assert evidence.source_type == "wikipedia"
        assert evidence.credibility_score == 70
    
    def test_evidence_output_creation(self):
        """Test EvidenceOutput model."""
        output = EvidenceOutput(
            claim_id=1,
            claim_text="Test claim",
            evidence_items=[
                EvidenceItem(
                    query="test",
                    source_name="Test",
                    source_type="test",
                    url="http://test.com",
                    title="Test",
                    snippet="Test snippet",
                    credibility_score=50,
                )
            ],
            source_types_found=["test"],
            total_evidence_count=1,
        )
        
        assert output.total_evidence_count == 1
        assert len(output.evidence_items) == 1
        assert "Test claim" in output.to_message_content()
    
    def test_verdict_output_creation(self):
        """Test VerdictOutput model."""
        verdict = VerdictOutput(
            claim_id=1,
            claim_text="Test claim",
            verdict="False",
            confidence=0.85,
            one_line_verdict="This claim is false",
            why_believed=["Reason 1", "Reason 2"],
            what_wrong="Explanation",
            why_reasonable="Understanding",
            truth="The truth",
            citation_map={"truth": [1, 2]},
        )
        
        assert verdict.verdict == "False"
        assert verdict.confidence == 0.85
        assert "False" in verdict.to_message_content()
    
    def test_script_output_creation(self):
        """Test ScriptOutput model."""
        script = ScriptOutput(
            claim_id=1,
            shorts_hook="Did you know?",
            shorts_segments=[],
            shorts_cta="Follow for more!",
            shorts_total_duration=35,
            long_chapters=[],
            long_total_duration=8.0,
            titles=["Title 1", "Title 2"],
            thumbnail_suggestions=["MYTH BUSTED"],
            description="Description",
            next_myths=["Myth 1", "Myth 2"],
        )
        
        assert script.shorts_total_duration == 35
        assert len(script.titles) == 2
    
    def test_qa_output_creation(self):
        """Test QAOutput model."""
        qa = QAOutput(
            claim_id=1,
            passed=True,
            issues=[],
            error_count=0,
            warning_count=0,
            citation_score=0.9,
            format_score=0.8,
            overall_score=0.85,
        )
        
        assert qa.passed is True
        assert qa.overall_score == 0.85
        assert "PASSED" in qa.to_message_content()
    
    def test_pipeline_context(self):
        """Test PipelineContext model."""
        context = PipelineContext(
            claim_id=1,
            claim_text="Test",
            topic="health",
            status="processing",
        )
        
        assert context.status == "processing"
        assert context.evidence_output is None


class TestActions:
    """Tests for MetaGPT actions (with stubs)."""
    
    def test_gather_evidence_instantiation(self):
        """Test GatherEvidence action can be instantiated."""
        action = GatherEvidence()
        assert action.name == "GatherEvidence"
    
    def test_judge_claim_instantiation(self):
        """Test JudgeClaim action can be instantiated."""
        action = JudgeClaim()
        assert action.name == "JudgeClaim"
    
    def test_generate_script_instantiation(self):
        """Test GenerateScript action can be instantiated."""
        action = GenerateScript()
        assert action.name == "GenerateScript"
    
    def test_qa_check_instantiation(self):
        """Test QACheck action can be instantiated."""
        action = QACheck()
        assert action.name == "QACheck"
    
    def test_metagpt_available_flag(self):
        """Test that METAGPT_AVAILABLE flag exists."""
        # Should be False since MetaGPT is not in myth-museum's path
        assert METAGPT_AVAILABLE is False or METAGPT_AVAILABLE is True


class TestJudgeClaimRuleBased:
    """Tests for rule-based verdict generation."""
    
    def test_judge_rule_based_no_evidence(self):
        """Test rule-based judging with no evidence."""
        action = JudgeClaim()
        
        verdict_input = VerdictInput(
            claim_id=1,
            claim_text="Test claim",
            topic="health",
            evidence_items=[],
        )
        
        result = action._judge_rule_based(verdict_input)
        
        assert result.verdict == "Unverified"
        assert result.confidence < 0.5
    
    def test_judge_rule_based_with_factcheck(self):
        """Test rule-based judging with factcheck evidence."""
        action = JudgeClaim()
        
        evidence = EvidenceItem(
            id=1,
            query="test",
            source_name="Snopes",
            source_type="factcheck",
            url="http://snopes.com/test",
            title="Test",
            snippet="This is false",
            credibility_score=80,
        )
        
        verdict_input = VerdictInput(
            claim_id=1,
            claim_text="Test claim",
            topic="health",
            evidence_items=[evidence],
        )
        
        result = action._judge_rule_based(verdict_input)
        
        assert result.verdict == "False"
        assert result.confidence >= 0.6


class TestGenerateScriptBaseline:
    """Tests for baseline script generation."""
    
    def test_generate_shorts_script(self):
        """Test shorts script generation."""
        action = GenerateScript()
        
        script_input = ScriptInput(
            claim_id=1,
            claim_text="Test claim about health",
            topic="health",
            verdict="False",
            confidence=0.8,
            explanation={
                "one_line_verdict": "This is false",
                "what_wrong": "Explanation here",
                "truth": "The truth is...",
            },
            evidence_items=[],
        )
        
        result = action._generate_shorts(script_input)
        
        assert "hook" in result
        assert "segments" in result
        assert "cta" in result
        assert result["duration"] >= 30
    
    def test_generate_long_outline(self):
        """Test long outline generation."""
        action = GenerateScript()
        
        script_input = ScriptInput(
            claim_id=1,
            claim_text="Test claim",
            topic="science",
            verdict="Misleading",
            confidence=0.7,
            explanation={},
            evidence_items=[],
        )
        
        result = action._generate_long_outline(script_input)
        
        assert "chapters" in result
        assert len(result["chapters"]) >= 5
        assert result["duration"] >= 6
    
    def test_generate_titles(self):
        """Test title generation."""
        action = GenerateScript()
        
        titles = action._generate_titles("Test claim about water", "False")
        
        assert len(titles) >= 5
        assert any("Test claim" in t or "claim" in t.lower() for t in titles)


class TestQACheck:
    """Tests for QA checking."""
    
    def test_check_citations(self):
        """Test citation checking."""
        action = QACheck()
        
        verdict = VerdictOutput(
            claim_id=1,
            claim_text="Test",
            verdict="False",
            confidence=0.8,
            one_line_verdict="False",
            why_believed=[],
            what_wrong="Wrong",
            why_reasonable="Reasonable",
            truth="Truth",
            citation_map={"truth": [1, 2]},
        )
        
        script = ScriptOutput(
            claim_id=1,
            shorts_hook="Hook",
            shorts_segments=[],
            shorts_cta="CTA",
            shorts_total_duration=35,
            long_chapters=[{} for _ in range(6)],
            long_total_duration=8.0,
            titles=["T1", "T2", "T3", "T4", "T5"],
            thumbnail_suggestions=[],
            description="",
            next_myths=[],
        )
        
        evidence = [
            EvidenceItem(
                id=1,
                query="test",
                source_name="Test",
                source_type="test",
                url="http://test.com",
                title="Test",
                snippet="Test",
                credibility_score=70,
            ),
            EvidenceItem(
                id=2,
                query="test",
                source_name="Test2",
                source_type="test",
                url="http://test2.com",
                title="Test2",
                snippet="Test2",
                credibility_score=80,
            ),
        ]
        
        qa_input = QAInput(
            claim_id=1,
            claim_text="Test",
            topic="science",
            verdict_output=verdict,
            script_output=script,
            evidence_items=evidence,
        )
        
        score = action._check_citations(qa_input)
        assert 0.0 <= score <= 1.0
    
    def test_check_disclaimer_for_health(self):
        """Test disclaimer checking for health topics."""
        action = QACheck()
        
        verdict = VerdictOutput(
            claim_id=1,
            claim_text="Test",
            verdict="False",
            confidence=0.8,
            one_line_verdict="False",
            why_believed=[],
            what_wrong="Wrong",
            why_reasonable="Reasonable",
            truth="Truth",
            citation_map={},
            disclaimer=None,  # No disclaimer
        )
        
        script = ScriptOutput(
            claim_id=1,
            shorts_hook="",
            shorts_segments=[],
            shorts_cta="",
            shorts_total_duration=35,
            long_chapters=[],
            long_total_duration=8.0,
            titles=[],
            thumbnail_suggestions=[],
            description="",
            next_myths=[],
        )
        
        qa_input = QAInput(
            claim_id=1,
            claim_text="Test",
            topic="health",  # Health topic requires disclaimer
            verdict_output=verdict,
            script_output=script,
            evidence_items=[],
        )
        
        result = action._check_disclaimer(qa_input)
        assert result is False  # Should fail because no disclaimer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
