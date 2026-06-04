"""測試比價 Agent"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.agents.comparison_agent import ComparisonAgent


class TestPriceGuardAgent:
    """測試 PriceGuardAgent"""

    @pytest.fixture
    def agent(self):
        """建立 PriceGuardAgent 實例"""
        from src.agents.price_guard import LLMPriceAnalysisOutput

        mock_llm = MagicMock()
        # 模擬 with_structured_output 返回的物件
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value=LLMPriceAnalysisOutput(
                is_valid=True,
                risk_level="low",
                risk_reason="價格合理",
                is_suspicious=False,
                purchase_reason="品質良好，價格合理",
                price_confidence=90
            )
        )
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        from src.agents.price_guard import PriceGuardAgent

        return PriceGuardAgent(llm=mock_llm)

    def test_detect_suspicious_text(self, agent):
        """測試可疑關鍵字掃描"""
        text = "這是一個高仿商品，拆機處理"
        keywords = agent.detect_suspicious_text(text)
        assert len(keywords) > 0
        assert any("高仿" in kw or "拆機" in kw for kw in keywords)

    @pytest.mark.asyncio
    async def test_analyze_with_llm(self, agent):
        """測試 LLM 分析"""
        text = "這是一個真實的商品，價格合理"
        price = 1000
        result = await agent.analyze_with_llm(text, price)

        assert result.is_valid is True
        assert result.risk_level == "low"
        assert result.purchase_reason is not None
        assert result.price_confidence == 90

    @pytest.mark.asyncio
    async def test_analyze_with_llm_timeout(self, agent):
        """測試 LLM 分析超時 - 測試 structured_llm.ainvoke 的超時"""
        # 設置超時異常
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
        agent.llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

        text = "測試文字"
        price = 1000

        # 由於 analyze_with_llm 有內部異常處理，會回退到規則判斷
        # 不會拋出 TimeoutError
        result = await agent.analyze_with_llm(text, price)
        # 應該回退到規則判斷，價格 1000 屬於低風險
        assert result.risk_level == "low"

    def test_check_price_risk_low_price(self, agent):
        """測試低價格風險判斷"""
        # 設定臨界值
        from src.core.config import settings

        settings.MIN_PRICE_THRESHOLD = 100
        settings.RISK_PRICE_MULTIPLIER = 0.5

        result = agent.check_price_risk(price=40)
        assert result.risk_level == "high"
        assert result.is_valid is False

    def test_check_price_risk_medium_price(self, agent):
        """測試中等價格風險判斷"""
        from src.core.config import settings

        settings.MIN_PRICE_THRESHOLD = 100

        result = agent.check_price_risk(price=80)
        assert result.risk_level == "medium"


class TestComparisonAgent:
    """測試 ComparisonAgent"""

    @pytest.fixture
    def agent(self):
        """建立 ComparisonAgent 實例（使用模擬 LLM）"""
        mock_llm = MagicMock()
        # 設定 model 屬性以通過 Pydantic 驗證
        type(mock_llm).model = "llama3"
        # 模擬 with_structured_output 返回的物件
        mock_structured_llm = MagicMock()
        mock_structured_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                real_price=32900.0,
                confidence=90,
                ranked_urls=["https://momo.tw/test1", "https://momo.tw/test2", "https://momo.tw/test3"]
            )
        )
        mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="測試結果"))
        return ComparisonAgent(llm_instance=mock_llm)

    def test_init(self, agent):
        """測試 Agent 初始化"""
        assert agent.llm is not None
        assert agent.price_guard is not None
        assert agent.scrape_agent is not None
        assert agent.cache is not None

    @pytest.mark.asyncio
    async def test_aenter_aexit(self, agent):
        """測試 Async Context Manager"""
        # 測試 __aenter__
        result = await agent.__aenter__()
        assert result is agent

        # 測試 __aexit__
        await agent.__aexit__(None, None, None)