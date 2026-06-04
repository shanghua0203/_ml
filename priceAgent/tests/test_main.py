"""測試主程式"""
import pytest
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from src.main import (
    print_header,
    run_comparison,
    _print_result,
    show_history,
    show_stats,
    clear_cache,
    run_single_scrape,
    safe_run,
    main,
)


class TestPrintHeader:
    """測試 print_header 函數"""

    def test_print_header(self, capsys):
        """測試印出標頭"""
        print_header()
        captured = capsys.readouterr()
        assert "=" * 60 in captured.out
        assert "本地端 AI 比價購物助理" in captured.out


class TestRunComparison:
    """測試 run_comparison 函數"""

    @pytest.mark.asyncio
    async def test_run_comparison_success(self):
        """測試成功執行比價"""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "product_name": "test",
            "recommendations": [],
            "llm_recommendation": "test",
        }

        with patch('src.agents.comparison_agent.ComparisonAgent') as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.__aenter__ = AsyncMock(return_value=mock_agent_instance)
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            mock_agent_instance.compare = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance

            with patch('src.main._print_result'):
                result = await run_comparison("test product")
                assert result == 0

    @pytest.mark.asyncio
    async def test_run_comparison_json_output(self):
        """測試 JSON 格式輸出"""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "product_name": "test",
            "recommendations": [],
            "llm_recommendation": "test",
        }

        with patch('src.agents.comparison_agent.ComparisonAgent') as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.__aenter__ = AsyncMock(return_value=mock_agent_instance)
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            mock_agent_instance.compare = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance

            with patch('sys.stdout'):
                result = await run_comparison("test product", output_json=True)
                assert result == 0

    @pytest.mark.asyncio
    async def test_run_comparison_validation_error(self):
        """測試設定驗證錯誤"""
        with patch('src.main.settings') as mock_settings:
            mock_settings.validate.return_value = ["BRAVE_API_KEY 未設定"]
            result = await run_comparison("test product")
            assert result == 1

    @pytest.mark.asyncio
    async def test_run_comparison_exception(self):
        """測試異常處理"""
        from src.agents.comparison_agent import ComparisonAgent

        # 模擬 ComparisonAgent 在進入時拋出異常
        with patch.object(ComparisonAgent, '__aenter__', side_effect=Exception("Test error")):
            result = await run_comparison("test product", output_json=True)
            assert result == 1


class TestPrintResult:
    """測試 _print_result 函數"""

    def test_print_result_with_recommendations(self, capsys):
        """測試印出結果 - 有推薦項"""
        mock_result = MagicMock()
        mock_result.most_recommended = {"name": "iPhone 15", "price": 32900, "source": "momo", "risk_level": "low", "safety_score": 95}
        mock_result.cheapest = {"name": "iPhone 15", "price": 29900, "source": "shopee"}
        mock_result.safest = {"name": "iPhone 15", "price": 32900, "source": "apple", "risk_level": "low", "safety_score": 100}
        mock_result.recommendations = [mock_result.most_recommended]
        mock_result.llm_recommendation = "推薦選擇官方旗艦店"

        with patch('src.main.print_header'):
            with patch('sys.stdout'):
                _print_result(mock_result)


class TestShowHistory:
    """測試 show_history 函數"""

    def test_show_history_empty(self, capsys):
        """測試空歷史紀錄"""
        with patch('src.utils.db.db') as mock_db:
            mock_db.get_history.return_value = AsyncMock(return_value=[])

            with patch('sys.stdout'):
                show_history()


class TestShowStats:
    """測試 show_stats 函數"""

    def test_show_stats(self, capsys):
        """測試統計資料"""
        with patch('src.utils.db.db') as mock_db:
            mock_db.get_stats.return_value = AsyncMock(return_value={"total_shopping_records": 10, "total_searches": 20})

            with patch('sys.stdout'):
                show_stats()


class TestClearCache:
    """測試 clear_cache 函數"""

    def test_clear_cache(self):
        """測試清除快取"""
        # 創建模擬快取
        mock_cache = MagicMock()
        mock_cache.clear_expired.return_value = 5

        # 在 src.main 模塊中 patch SearchCache
        with patch('src.main.SearchCache', return_value=mock_cache) as MockCache:
            result = clear_cache()

            # 驗證 SearchCache 被調用
            MockCache.assert_called_once()
            # 驗證 clear_expired 被調用
            mock_cache.clear_expired.assert_called_once()
            # clear_cache 不返回值
            assert result is None


class TestRunSingleScrape:
    """測試 run_single_scrape 函數"""

    @pytest.mark.asyncio
    async def test_run_single_scrape_success(self):
        """測試成功抓取網頁"""
        mock_result = MagicMock()
        mock_result.title = "Test Product"
        mock_result.markdown = "# Test Product\nThis is a test product."
        mock_result.scraped_at = "2024-01-01"

        with patch('src.agents.scrape_agent.ScrapeAgent') as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.__aenter__ = AsyncMock(return_value=mock_agent_instance)
            mock_agent_instance.__aexit__ = AsyncMock(return_value=None)
            mock_agent_instance.scrape = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent_instance

            result = await run_single_scrape("https://example.com")
            assert result is None  # run_single_scrape 不返回值


class TestSafeRun:
    """測試 safe_run 函數"""

    def test_safe_run(self):
        """測試安全執行"""
        async def test_coro():
            return "success"

        result = safe_run(test_coro())
        assert result == "success"


class TestMain:
    """測試 main 函數"""

    def test_main_with_search(self):
        """測試搜尋參數"""
        with patch('sys.argv', ['main.py', '--search', 'iphone 15']):
            with patch('src.main.run_comparison') as mock_compare:
                mock_compare.return_value = 0
                result = main()
                assert result == 0

    def test_main_with_json(self):
        """測試 JSON 參數"""
        with patch('sys.argv', ['main.py', '--search', 'iphone 15', '--json']):
            with patch('src.main.run_comparison') as mock_compare:
                mock_compare.return_value = 0
                result = main()
                assert result == 0

    def test_main_with_history(self):
        """測試歷史參數"""
        with patch('sys.argv', ['main.py', '--history']):
            with patch('src.main.show_history') as mock_history:
                result = main()
                assert result == 0
                mock_history.assert_called_once()

    def test_main_with_stats(self):
        """測試統計參數"""
        with patch('sys.argv', ['main.py', '--stats']):
            with patch('src.main.show_stats') as mock_stats:
                result = main()
                assert result == 0
                mock_stats.assert_called_once()

    def test_main_with_scrape(self):
        """測試抓取參數"""
        with patch('sys.argv', ['main.py', '--scrape', 'https://example.com']):
            with patch('src.main.run_single_scrape') as mock_scrape:
                mock_scrape.return_value = 0
                result = main()
                assert result == 0

    def test_main_with_clear_cache(self):
        """測試清除快取參數"""
        with patch('sys.argv', ['main.py', '--clear-cache']):
            with patch('src.main.clear_cache') as mock_clear:
                result = main()
                assert result == 0
                mock_clear.assert_called_once()

    def test_main_with_env(self):
        """測試環境變數參數"""
        with patch('sys.argv', ['main.py', '--env']):
            with patch('src.main.settings') as mock_settings:
                mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_settings.OLLAMA_MODEL = "gemma3:27b"
                mock_settings.BRAVE_API_KEY = "test_key"
                result = main()
                assert result == 0

    def test_main_with_no_args(self):
        """測試無參數"""
        with patch('sys.argv', ['main.py']):
            with patch('argparse.ArgumentParser.print_help'):
                result = main()
                assert result == 0
