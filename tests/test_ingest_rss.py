"""
Tests for ingest.rss module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ingest.rss import fetch_feed, fetch_full_text


class TestFetchFeed:
    """Tests for fetch_feed function."""
    
    @pytest.mark.asyncio
    async def test_fetch_feed_mock(self):
        """Test fetching a feed with mocked response."""
        mock_feed_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Article 1</title>
                    <link>https://example.com/article1</link>
                    <description>This is a test article.</description>
                </item>
                <item>
                    <title>Test Article 2</title>
                    <link>https://example.com/article2</link>
                    <description>This is another test article.</description>
                </item>
            </channel>
        </rss>
        """
        
        with patch("ingest.rss.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            
            mock_response = MagicMock()
            mock_response.text = mock_feed_content
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            entries = await fetch_feed("https://example.com/feed.xml")
            
            assert len(entries) == 2
            assert entries[0]["title"] == "Test Article 1"
            assert entries[0]["link"] == "https://example.com/article1"
    
    @pytest.mark.asyncio
    async def test_fetch_feed_empty(self):
        """Test handling of empty feed."""
        mock_feed_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Empty Feed</title>
            </channel>
        </rss>
        """
        
        with patch("ingest.rss.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            
            mock_response = MagicMock()
            mock_response.text = mock_feed_content
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            entries = await fetch_feed("https://example.com/empty.xml")
            
            assert len(entries) == 0


class TestFetchFullText:
    """Tests for fetch_full_text function."""
    
    @pytest.mark.asyncio
    async def test_fetch_full_text_article(self):
        """Test extracting content from article HTML."""
        mock_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test</title></head>
        <body>
            <nav>Navigation</nav>
            <article>
                <h1>Article Title</h1>
                <p>This is the main content of the article. It contains important information.</p>
                <p>More content here with details and explanations about the topic.</p>
            </article>
            <footer>Footer content</footer>
        </body>
        </html>
        """
        
        with patch("ingest.rss.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            
            mock_response = MagicMock()
            mock_response.text = mock_html
            mock_response.raise_for_status = MagicMock()
            
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            
            content = await fetch_full_text("https://example.com/article")
            
            assert content is not None
            assert "Article Title" in content
            assert "main content" in content
            assert "Navigation" not in content  # Should be removed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
