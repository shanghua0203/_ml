"""測試資料庫操作"""
import pytest
import os
import tempfile
from pathlib import Path

from src.utils.db import Database


@pytest.fixture
def temp_db():
    """建立臨時資料庫"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = Database(db_path=db_path)
        yield db


class TestDatabase:
    """測試 Database 類別"""

    @pytest.mark.asyncio
    async def test_save_shopping_result(self, temp_db):
        """測試儲存購物結果"""
        result_id = await temp_db.save_shopping_result(
            query="iphone 15",
            product_name="iPhone 15",
            product_url="https://example.com/iphone15",
            source_platform="momo",
            price=32900,
            risk_level="low",
            is_recommendation=True,
        )

        assert result_id > 0

    @pytest.mark.asyncio
    async def test_save_search_log(self, temp_db):
        """測試儲存搜尋日誌"""
        result_id = await temp_db.save_search_log(query="macbook pro", results_count=5)

        assert result_id > 0

    @pytest.mark.asyncio
    async def test_get_history(self, temp_db):
        """測試取得歷史紀錄"""
        # 先儲存一些資料
        await temp_db.save_shopping_result(
            query="iphone 15",
            product_name="iPhone 15",
            product_url="https://example.com/iphone15",
            source_platform="momo",
            price=32900,
            risk_level="low",
        )

        await temp_db.save_shopping_result(
            query="macbook pro",
            product_name="MacBook Pro",
            product_url="https://example.com/macbook",
            source_platform="pchome",
            price=61900,
            risk_level="low",
        )

        # 取得歷史紀錄
        history = await temp_db.get_history(limit=10)

        assert len(history) == 2
        # 驗證有正確儲存兩筆資料
        queries = [h["search_query"] for h in history]
        assert "macbook pro" in queries
        assert "iphone 15" in queries

    @pytest.mark.asyncio
    async def test_get_history_by_query(self, temp_db):
        """測試根據關鍵字取得歷史紀錄"""
        # 先儲存一些資料
        await temp_db.save_shopping_result(
            query="iphone 15",
            product_name="iPhone 15",
            product_url="https://example.com/iphone15",
            source_platform="momo",
            price=32900,
            risk_level="low",
        )

        await temp_db.save_shopping_result(
            query="iphone 14",
            product_name="iPhone 14",
            product_url="https://example.com/iphone14",
            source_platform="shopee",
            price=25900,
            risk_level="low",
        )

        # 根據關鍵字搜尋
        history = await temp_db.get_history(query="iphone 15", limit=10)

        assert len(history) == 1
        assert "iphone 15" in history[0]["search_query"]

    @pytest.mark.asyncio
    async def test_get_stats(self, temp_db):
        """測試取得統計資料"""
        # 先儲存一些資料
        await temp_db.save_shopping_result(
            query="iphone 15",
            product_name="iPhone 15",
            product_url="https://example.com/iphone15",
            source_platform="momo",
            price=32900,
            risk_level="low",
        )

        await temp_db.save_search_log(query="macbook pro", results_count=5)
        await temp_db.save_search_log(query="airpods", results_count=3)

        # 取得統計
        stats = await temp_db.get_stats()

        assert stats["total_shopping_records"] == 1
        assert stats["total_searches"] == 2


class TestDatabaseEdgeCases:
    """測試資料庫邊界情況"""

    @pytest.fixture
    def temp_db(self):
        """建立臨時資料庫"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_edge.db"
            db = Database(db_path=db_path)
            yield db

    @pytest.mark.asyncio
    async def test_empty_history(self, temp_db):
        """測試空的歷史紀錄"""
        history = await temp_db.get_history(limit=10)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_get_recent_searches(self, temp_db):
        """測試取得最近搜尋"""
        await temp_db.save_search_log(query="iphone 15", results_count=5)
        await temp_db.save_search_log(query="macbook pro", results_count=3)

        searches = await temp_db.get_recent_searches(limit=2)

        assert len(searches) == 2
        # 驗證有正確儲存兩個關鍵字
        assert "macbook pro" in [s.lower() for s in searches]
        assert "iphone 15" in [s.lower() for s in searches]