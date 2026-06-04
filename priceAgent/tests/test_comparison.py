"""測試比價流程"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.comparison_agent import ComparisonAgent
from src.models.products import ComparisonResult


class TestComparisonFlow:
    """測試完整比價流程"""

    @pytest.fixture
    def mock_agent(self):
        """建立模擬的 ComparisonAgent"""
        mock_llm = MagicMock()
        type(mock_llm).model = "llama3"
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="測試結果"))

        agent = ComparisonAgent(llm_instance=mock_llm)

        # 模擬搜尋結果
        agent._search_products = AsyncMock(
            return_value=[
                {
                    "title": "iPhone 15 官方旗艦店",
                    "url": "https://momo.tw/iphone15",
                    "description": "iPhone 15 官方旗艦店，全新未拆封",
                    "position": 1,
                },
                {
                    "title": "iPhone 15 特價",
                    "url": "https://shopee.tw/iphone15",
                    "description": "iPhone 15 限時特價",
                    "position": 2,
                },
                {
                    "title": "iPhone 15 詳細資訊",
                    "url": "https://apple.com/tw/iphone15",
                    "description": "Apple 官方網站 - iPhone 15 詳細規格",
                    "position": 3,
                },
            ]
        )

        # 模擬 LLM 排序
        agent._rank_results_with_llm = AsyncMock(
            return_value=[
                {
                    "title": "iPhone 15 官方旗艦店",
                    "url": "https://momo.tw/iphone15",
                    "description": "iPhone 15 官方旗艦店，全新未拆封",
                    "position": 1,
                },
                {
                    "title": "iPhone 15 特價",
                    "url": "https://shopee.tw/iphone15",
                    "description": "iPhone 15 限時特價",
                    "position": 2,
                },
            ]
        )

        # 模擬網頁抓取
        agent._scrape_with_retry = AsyncMock(
            return_value={
                "url": "https://momo.tw/iphone15",
                "markdown": "## iPhone 15\n售價：NT$32,900\n全新未拆封",
                "title": "iPhone 15 官方旗艦店",
            }
        )

        # 模擬價格提取
        agent._extract_price_from_text = MagicMock(return_value=32900)

        # 模擬 LLM 價格提取
        agent._extract_price_with_llm = AsyncMock(return_value=(32900, 90))

        # 模擬 LLM 分析
        agent.price_guard.analyze_with_llm = AsyncMock(
            return_value=MagicMock(
                is_valid=True,
                risk_level="low",
                risk_reason=None,
                is_suspicious=False,
                purchase_reason="品質良好，價格合理",
                price_confidence=90,
            )
        )

        # 模擬 LLM 建議
        agent._generate_llm_recommendation = AsyncMock(
            return_value="推薦選擇 iPhone 15 官方旗艦店，安全分數高。"
        )

        return agent

    @pytest.mark.asyncio
    async def test_compare_with_mocked_api(self, mock_agent):
        """測試模擬 API 的比價流程"""
        result = await mock_agent.compare("iphone 15")

        # 驗證結果
        assert isinstance(result, ComparisonResult)
        assert result.product_name == "iphone 15"
        assert len(result.recommendations) > 0
        assert result.llm_recommendation is not None


class TestEdgeCases:
    """測試邊界情況"""

    @pytest.fixture
    def mock_agent(self):
        """建立模擬的 ComparisonAgent"""
        mock_llm = MagicMock()
        type(mock_llm).model = "llama3"
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="測試結果"))
        return ComparisonAgent(llm_instance=mock_llm)

    @pytest.mark.asyncio
    async def test_empty_results_handling(self, mock_agent):
        """測試空搜尋結果"""
        mock_agent._search_products = AsyncMock(return_value=[])

        result = await mock_agent.compare("nonexistent product")
        assert isinstance(result, ComparisonResult)
        assert len(result.recommendations) == 0
        assert "搜尋失敗" in result.llm_recommendation

    @pytest.mark.asyncio
    async def test_api_error_handling(self, mock_agent):
        """測試 API 錯誤處理"""
        mock_agent._search_products = AsyncMock(
            side_effect=RuntimeError("BRAVE_API_KEY 未設定")
        )

        result = await mock_agent.compare("test product")
        assert isinstance(result, ComparisonResult)
        assert len(result.recommendations) == 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_agent):
        """測試超時處理"""
        mock_agent._search_products = AsyncMock(
            return_value=[
                {
                    "title": "測試商品",
                    "url": "https://example.com",
                    "description": "測試",
                    "position": 1,
                }
            ]
        )
        mock_agent._rank_results_with_llm = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_agent._scrape_with_retry = AsyncMock(return_value=None)

        result = await mock_agent.compare("test product")
        assert isinstance(result, ComparisonResult)