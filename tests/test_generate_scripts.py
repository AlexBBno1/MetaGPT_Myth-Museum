"""
Tests for pipeline.generate_scripts module.
"""

import pytest
import tempfile
from pathlib import Path

from pipeline.generate_scripts import (
    generate_long_outline_baseline,
    generate_shorts_script_baseline,
    generate_titles,
    generate_next_myths,
    export_packet_json,
    export_packet_md,
)
from core.models import Packet, Verdict


class TestGenerateShortsScript:
    """Tests for generate_shorts_script_baseline function."""
    
    @pytest.fixture
    def sample_verdict(self):
        """Create a sample verdict."""
        return Verdict(
            id=1,
            claim_id=1,
            verdict="False",
            explanation_json={
                "one_line_verdict": "This claim is false.",
                "what_wrong": "The claim oversimplifies complex research.",
                "truth": "The reality is more nuanced.",
            },
            confidence=0.8,
        )
    
    def test_shorts_script_has_hook(self, sample_verdict):
        """Test that shorts script has a hook."""
        script = generate_shorts_script_baseline(sample_verdict, "Test claim")
        
        assert "hook" in script
        assert len(script["hook"]) > 0
    
    def test_shorts_script_has_segments(self, sample_verdict):
        """Test that shorts script has segments."""
        script = generate_shorts_script_baseline(sample_verdict, "Test claim")
        
        assert "segments" in script
        assert len(script["segments"]) > 0
        
        for segment in script["segments"]:
            assert "time_start" in segment
            assert "time_end" in segment
            assert "narration" in segment
    
    def test_shorts_script_has_cta(self, sample_verdict):
        """Test that shorts script has CTA."""
        script = generate_shorts_script_baseline(sample_verdict, "Test claim")
        
        assert "cta" in script
        assert len(script["cta"]) > 0


class TestGenerateLongOutline:
    """Tests for generate_long_outline_baseline function."""
    
    @pytest.fixture
    def sample_verdict(self):
        """Create a sample verdict."""
        return Verdict(
            id=1,
            claim_id=1,
            verdict="Misleading",
            explanation_json={
                "one_line_verdict": "This claim is misleading.",
                "why_reasonable": "It's based on outdated research.",
            },
            confidence=0.7,
        )
    
    def test_outline_has_chapters(self, sample_verdict):
        """Test that outline has chapters."""
        outline = generate_long_outline_baseline(sample_verdict, "Test claim")
        
        assert "chapters" in outline
        assert len(outline["chapters"]) >= 5
    
    def test_chapters_have_required_fields(self, sample_verdict):
        """Test that chapters have required fields."""
        outline = generate_long_outline_baseline(sample_verdict, "Test claim")
        
        for chapter in outline["chapters"]:
            assert "chapter_number" in chapter
            assert "title" in chapter
            assert "duration_minutes" in chapter
            assert "key_points" in chapter


class TestGenerateTitles:
    """Tests for generate_titles function."""
    
    def test_generates_multiple_titles(self):
        """Test that multiple titles are generated."""
        titles = generate_titles("Drinking water is good", "False")
        
        assert len(titles) >= 5
    
    def test_titles_contain_claim(self):
        """Test that titles reference the claim."""
        claim = "Drinking water"
        titles = generate_titles(claim, "False")
        
        # At least some titles should contain the claim
        assert any(claim[:10] in t for t in titles)


class TestExportPacket:
    """Tests for export functions."""
    
    @pytest.fixture
    def sample_packet(self):
        """Create a sample packet."""
        return Packet(
            id=1,
            claim_id=42,
            packet_json={
                "claim": "Test claim",
                "verdict": "False",
                "confidence": 0.8,
                "topic": "health",
                "one_line_verdict": "This is false.",
                "why_believed": ["Reason 1", "Reason 2"],
                "what_wrong": "Explanation",
                "why_reasonable": "Understandable",
                "truth": "The truth is...",
                "sources": [],
                "created_at": "2024-01-01T00:00:00",
            },
        )
    
    def test_export_json(self, sample_packet):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            file_path = export_packet_json(sample_packet, output_dir)
            
            assert file_path.exists()
            assert file_path.name == "42.json"
    
    def test_export_md(self, sample_packet):
        """Test Markdown export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            file_path = export_packet_md(sample_packet, output_dir)
            
            assert file_path.exists()
            assert file_path.name == "42.md"
            
            content = file_path.read_text()
            assert "Test claim" in content
            assert "False" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
