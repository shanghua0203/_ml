"""測試爬蟲 Agent"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.scrape_agent import ScrapeAgent, ScrapeResult


class TestScrapeAgent:
    """測試 ScrapeAgent"""

    @pytest.fixture
    def agent(self):
        """建立 ScrapeAgent 實例"""
        return ScrapeAgent()

    def test_init(self, agent):
        """測試 Agent 初始化"""
        assert agent.config is not None
        assert agent.cache is not None
        assert agent._crawler is None

    @pytest.mark.asyncio
    async def test_aenter_aexit(self, agent):
        """測試 Async Context Manager"""
        # 測試 __aenter__
        result = await agent.__aenter__()
        assert result is agent
        assert agent._crawler is not None

        # 測試 __aexit__
        await agent.__aexit__(None, None, None)
        assert agent._crawler is None

    @pytest.mark.asyncio
    async def test_scrape_with_valid_url(self, agent):
        """測試有效 URL 的抓取"""
        # 模擬抓取結果
        mock_result = MagicMock()
        mock_result.markdown = "# 測試商品\n這是一個測試商品"
        mock_result.title = "測試商品"
        mock_result.images = ["https://example.com/img.jpg"]
        mock_result.links = ["https://example.com/link"]

        with patch.object(agent, "_crawler"):
            agent._crawler.arun = AsyncMock(return_value=mock_result)

            result = await agent.scrape("https://example.com")

            assert result.url == "https://example.com"
            assert "# 測試商品" in result.markdown
            assert result.title == "測試商品"