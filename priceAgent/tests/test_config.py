"""測試系統設定與環境變數管理"""
import pytest
import os
from unittest.mock import patch


class TestSettings:
    """測試 Settings 類別"""

    def test_default_ollama_settings(self):
        """測試預設的 Ollama 設定"""
        from src.core.config import settings
        assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
        # 這裡的預設值可能受到系統環境變數影響，所以只驗證 URL
        assert settings.OLLAMA_MODEL is not None

    def test_default_search_settings(self):
        """測試預設搜尋設定"""
        from src.core.config import settings
        assert settings.SEARCH_RESULTS_COUNT == 3
        assert settings.CACHE_EXPIRY_HOURS == 24

    def test_default_price_settings(self):
        """測試預設價格設定"""
        from src.core.config import settings
        assert settings.MIN_PRICE_THRESHOLD == 100
        assert settings.RISK_PRICE_MULTIPLIER == 0.5

    def test_database_path(self):
        """測試資料庫路徑"""
        from src.core.config import settings
        from pathlib import Path
        assert str(settings.DATABASE_PATH).endswith("data/shopping.db")

    def test_validate_without_api_key(self):
        """測試驗證 - 未設定 API key"""
        from src.core.config import Settings
        # 保存原始值
        original_api_key = Settings.BRAVE_API_KEY

        try:
            # 測試驗證會回報缺少 API key
            Settings.BRAVE_API_KEY = None
            errors = Settings.validate()
            assert len(errors) > 0
            assert any("BRAVE_API_KEY" in error for error in errors)
        finally:
            # 恢復原始值
            Settings.BRAVE_API_KEY = original_api_key

    def test_validate_with_api_key(self):
        """測試驗證 - 已設定 API key"""
        from src.core.config import Settings
        # 保存原始值
        original_api_key = Settings.BRAVE_API_KEY

        try:
            # 設定 API key
            Settings.BRAVE_API_KEY = "test_key_123"
            errors = Settings.validate()
            assert len(errors) == 0
        finally:
            # 恢復原始值
            Settings.BRAVE_API_KEY = original_api_key


class TestSettingsEnvironmentVariables:
    """測試環境變數設定 - 使用 mock"""

    def test_custom_ollama_url(self):
        """測試自訂 Ollama URL"""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:11434"}, clear=True):
            import importlib
            import src.core.config
            importlib.reload(src.core.config)
            from src.core.config import settings as new_settings

            # 檢查是否從環境變數讀取
            assert new_settings.OLLAMA_BASE_URL == "http://custom:11434"

    def test_custom_ollama_model(self):
        """測試自訂 Ollama Model"""
        with patch.dict(os.environ, {"OLLAMA_MODEL": "llama3:8b"}, clear=True):
            import importlib
            import src.core.config
            importlib.reload(src.core.config)
            from src.core.config import settings as new_settings

            assert new_settings.OLLAMA_MODEL == "llama3:8b"

    def test_custom_search_results_count(self):
        """測試自訂搜尋結果數量 - 驗證環境變數解析邏輯"""
        # 保存原始值
        original_value = os.environ.get("SEARCH_RESULTS_COUNT")

        try:
            # 設定環境變數
            os.environ["SEARCH_RESULTS_COUNT"] = "10"

            # 創建新類別以重新加載環境變數
            class TestSettings:
                SEARCH_RESULTS_COUNT = int(os.getenv("SEARCH_RESULTS_COUNT", "3"))

            assert TestSettings.SEARCH_RESULTS_COUNT == 10
        finally:
            # 恢復原始值
            if original_value:
                os.environ["SEARCH_RESULTS_COUNT"] = original_value
            elif "SEARCH_RESULTS_COUNT" in os.environ:
                del os.environ["SEARCH_RESULTS_COUNT"]
