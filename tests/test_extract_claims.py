"""
Tests for pipeline.extract_claims module.
"""

import pytest

from pipeline.extract_claims import (
    classify_topic,
    extract_claims_regex,
    score_claim,
)
from core.constants import TopicEnum


class TestExtractClaimsRegex:
    """Tests for extract_claims_regex function."""
    
    def test_extract_simple_claim(self):
        """Test extracting a simple claim."""
        text = "People say drinking 8 glasses of water daily is necessary for health."
        claims = extract_claims_regex(text)
        
        assert len(claims) >= 1
        assert any("water" in c.lower() for c in claims)
    
    def test_extract_multiple_claims(self):
        """Test extracting multiple claims."""
        text = """
        Some believe vaccines cause autism. 
        It is claimed that the moon landing was faked.
        Studies show that breakfast is the most important meal.
        """
        claims = extract_claims_regex(text)
        
        assert len(claims) <= 3  # Max 3 claims
    
    def test_empty_text(self):
        """Test with empty text."""
        claims = extract_claims_regex("")
        assert claims == []
    
    def test_no_claims(self):
        """Test with text containing no claims."""
        text = "The weather is nice today. I had lunch at noon."
        claims = extract_claims_regex(text)
        # May or may not find claims, just shouldn't crash
        assert isinstance(claims, list)


class TestClassifyTopic:
    """Tests for classify_topic function."""
    
    def test_health_topic(self):
        """Test health topic classification."""
        claim = "Drinking water helps prevent disease and improves health."
        topic = classify_topic(claim)
        assert topic == TopicEnum.HEALTH
    
    def test_history_topic(self):
        """Test history topic classification."""
        claim = "Napoleon was actually quite short compared to other historical figures."
        topic = classify_topic(claim)
        assert topic == TopicEnum.HISTORY
    
    def test_science_topic(self):
        """Test science topic classification."""
        claim = "Evolution explains how species develop through natural selection."
        topic = classify_topic(claim)
        assert topic == TopicEnum.SCIENCE
    
    def test_unknown_topic(self):
        """Test unknown topic classification."""
        claim = "This is just a random statement."
        topic = classify_topic(claim)
        assert topic == TopicEnum.UNKNOWN


class TestScoreClaim:
    """Tests for score_claim function."""
    
    def test_score_range(self):
        """Test that score is in valid range."""
        claim = "This is a test claim about health myths."
        score = score_claim(claim)
        assert 0 <= score <= 100
    
    def test_viral_keywords_boost(self):
        """Test that viral keywords boost score."""
        claim1 = "Something about health"
        claim2 = "MYTH BUSTED: The truth about health revealed"
        
        score1 = score_claim(claim1)
        score2 = score_claim(claim2)
        
        assert score2 > score1
    
    def test_numbers_boost(self):
        """Test that numbers boost score."""
        claim1 = "Drinking water is good"
        claim2 = "Drinking 8 glasses of water is good"
        
        score1 = score_claim(claim1)
        score2 = score_claim(claim2)
        
        assert score2 >= score1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
