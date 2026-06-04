"""測試快取機制"""
import pytest
import asyncio
import time
from pathlib import Path
import tempfile

from src.core.cache import SearchCache


class TestSearchCache:
    """測試 SearchCache"""

    @pytest.fixture
    def cache(self):
        """建立 SearchCache 實例"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "cache.db"
            cache = SearchCache(db_path=db_path)
            yield cache

    def test_set_and_get(self, cache):
        """測試快取設定與取得"""
        query = "iphone 15"
        results = [{"title": "iPhone 15", "url": "https://example.com"}]

        cache.set(query, results)
        cached = cache.get(query)

        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["title"] == "iPhone 15"

    def test_get_nonexistent(self, cache):
        """測試取得不存在的快取"""
        result = cache.get("nonexistent query")
        assert result is None

    def test_cache_expiry(self, cache):
        """測試快取過期"""
        query = "test query"
        results = [{"title": "Test", "url": "https://example.com"}]

        # 設定快取
        cache.set(query, results)

        # 手動將過期時間設為過去
        import sqlite3
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute(
                "UPDATE search_cache SET expires_at = datetime('now', '-1 hour') WHERE query_hash = ?",
                (cache._generate_hash(query),)
            )
            conn.commit()

        # 應該無法取得過期快取
        result = cache.get(query)
        assert result is None

    def test_clear_expired(self, cache):
        """測試清除過期快取"""
        query1 = "expired query"
        query2 = "valid query"
        results = [{"title": "Test", "url": "https://example.com"}]

        # 設定兩個快取
        cache.set(query1, results)
        cache.set(query2, results)

        # 手動將一個設為過期
        import sqlite3
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute(
                "UPDATE search_cache SET expires_at = datetime('now', '-1 hour') WHERE query_hash = ?",
                (cache._generate_hash(query1),)
            )
            conn.commit()

        # 清除過期快取
        count = cache.clear_expired()

        assert count == 1

        # 過期快取應被清除
        assert cache.get(query1) is None
        # 有效快取應保留
        assert cache.get(query2) is not None

    def test_generate_hash(self, cache):
        """測試 Hash 產生"""
        hash1 = cache._generate_hash("test query")
        hash2 = cache._generate_hash("test query")
        hash3 = cache._generate_hash("different query")

        # 相同查詢應產生相同 hash
        assert hash1 == hash2
        # 不同查詢應產生不同 hash
        assert hash1 != hash3

    def test_cache_is_shopping_site_filter(self, cache):
        """測試快取包含購物網站過濾"""
        query = "macbook pro"
        results = [
            {"title": "MacBook Pro - Apple", "url": "https://apple.com", "description": "Apple 官方網站", "position": 1},
            {"title": "MacBook Pro - PChome", "url": "https://pchome.com", "description": "PChome 購物中心", "position": 2},
            {"title": "MacBook Pro 評測", "url": "https://techbang.com", "description": "技術社群評測", "position": 3},
        ]

        cache.set(query, results)
        cached = cache.get(query)

        assert cached is not None
        # 快取應包含原始結果
        assert len(cached) == 3