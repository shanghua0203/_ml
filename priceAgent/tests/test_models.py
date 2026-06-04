"""測試 Pydantic 資料模型"""
import pytest
from src.models.products import (
    PriceAnalysis,
    FraudDetectionResult,
    Product,
    SearchResult,
    ScrapeResult,
)


class TestPriceAnalysis:
    """測試 PriceAnalysis 模型"""

    def test_price_analysis_with_all_fields(self):
        """測試所有欄位都提供的 PriceAnalysis"""
        analysis = PriceAnalysis(
            is_valid=True,
            price=1000,
            risk_level="low",
            is_suspicious=False,
            purchase_reason="品質良好，價格合理",
            price_confidence=85,
        )
        assert analysis.is_valid is True
        assert analysis.price == 1000
        assert analysis.risk_level == "low"
        assert analysis.purchase_reason == "品質良好，價格合理"
        assert analysis.price_confidence == 85

    def test_price_analysis_minimal_fields(self):
        """測試只提供必要欄位的 PriceAnalysis"""
        analysis = PriceAnalysis(
            is_valid=True,
            price=1000,
            risk_level="medium",
            is_suspicious=True,
        )
        assert analysis.is_valid is True
        assert analysis.price == 1000
        assert analysis.risk_level == "medium"
        assert analysis.is_suspicious is True
        assert analysis.purchase_reason is None
        assert analysis.price_confidence == 0  # 預設值

    def test_price_analysis_invalid_risk_level(self):
        """測試無效的風險等級"""
        with pytest.raises(ValueError):
            PriceAnalysis(
                is_valid=True,
                price=1000,
                risk_level="invalid",
                is_suspicious=False,
            )


class TestFraudDetectionResult:
    """測試 FraudDetectionResult 模型"""

    def test_fraud_detection_with_keywords(self):
        """測試有可疑關鍵字的 FraudDetectionResult"""
        result = FraudDetectionResult(
            url="https://example.com",
            has_suspicious_text=True,
            suspicious_keywords=["高仿", "拆機"],
        )
        assert result.has_suspicious_text is True
        assert len(result.suspicious_keywords) == 2

    def test_fraud_detection_no_keywords(self):
        """測試沒有可疑關鍵字的 FraudDetectionResult"""
        result = FraudDetectionResult(
            url="https://example.com",
            has_suspicious_text=False,
            suspicious_keywords=[],
        )
        assert result.has_suspicious_text is False
        assert len(result.suspicious_keywords) == 0


class TestProduct:
    """測試 Product 模型"""

    def test_product_with_all_fields(self):
        """測試所有欄位都提供的 Product"""
        product = Product(
            name="iPhone 15",
            price=32900,
            url="https://example.com/iphone15",
            source="momo",
            original_price=35900,
            discount_rate=0.9,
        )
        assert product.name == "iPhone 15"
        assert product.price == 32900
        assert product.source == "momo"
        assert product.original_price == 35900
        assert product.discount_rate == 0.9

    def test_product_minimal_fields(self):
        """測試只提供必要欄位的 Product"""
        product = Product(
            name="MacBook Pro",
            price=61900,
            url="https://example.com/macbook",
            source="pchome",
        )
        assert product.name == "MacBook Pro"
        assert product.price == 61900
        assert product.original_price is None
        assert product.discount_rate is None


class TestSearchResult:
    """測試 SearchResult 模型"""

    def test_search_result(self):
        """測試 SearchResult"""
        result = SearchResult(
            title="iPhone 15 詳細資訊",
            url="https://example.com/iphone15",
            description="最新款 iPhone 15，配備 A17 處理器。",
            position=1,
        )
        assert result.title == "iPhone 15 詳細資訊"
        assert result.position == 1


class TestScrapeResult:
    """測試 ScrapeResult 模型"""

    def test_scrape_result(self):
        """測試 ScrapeResult"""
        result = ScrapeResult(
            url="https://example.com",
            markdown="# 商品說明\n這是一個測試商品。",
            title="測試商品",
            images=["https://example.com/img1.jpg"],
            links=["https://example.com/link1"],
        )
        assert result.url == "https://example.com"
        assert "# 商品說明" in result.markdown
        assert len(result.images) == 1
        assert len(result.links) == 1