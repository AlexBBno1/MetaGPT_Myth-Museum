"""
Tests for pipeline.build_evidence module.
"""

import pytest

from pipeline.build_evidence import generate_queries


class TestGenerateQueries:
    """Tests for generate_queries function."""
    
    def test_generates_multiple_queries(self):
        """Test that multiple queries are generated."""
        claim = "Drinking 8 glasses of water daily is necessary for health."
        queries = generate_queries(claim)
        
        assert len(queries) >= 2
        assert len(queries) <= 6
    
    def test_includes_original_claim(self):
        """Test that original claim is included."""
        claim = "Test claim about health"
        queries = generate_queries(claim)
        
        assert claim in queries
    
    def test_includes_fact_check_query(self):
        """Test that fact check query is included."""
        claim = "Drinking water is important"
        queries = generate_queries(claim)
        
        assert any("fact check" in q.lower() for q in queries)
    
    def test_handles_chinese_claims(self):
        """Test handling of Chinese claims."""
        claim = "每天喝八杯水是必要的"
        queries = generate_queries(claim)
        
        assert len(queries) >= 2
        # Should include Chinese-specific queries
        assert any("事實查核" in q or "迷思" in q for q in queries)
    
    def test_empty_claim(self):
        """Test with empty claim."""
        queries = generate_queries("")
        
        assert len(queries) >= 1  # At least the original (empty) claim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
