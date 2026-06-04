"""
================================================================================
搜尋結果快取機制模組
================================================================================

本模組提供搜尋結果的快取功能，
避免在 24 小時內重複呼叫 API，節省資源並提高效能。

主要功能：
1. SearchCache 類別 - 管理搜尋結果快取
   - get() - 取得快取的搜尋結果
   - set() - 儲存搜尋結果到快取
   - clear_expired() - 清除過期快取
   - cleanup() - 清理快取（呼叫 clear_expired 並顯示訊息）

使用方式：
---------
from src.core.cache import SearchCache

cache = SearchCache()

# 儲存搜尋結果
cache.set("iphone 15", [{"title": "iPhone 15", "url": "..."}])

# 取得快取結果
cached = cache.get("iphone 15")
if cached:
    print("使用快取結果")
else:
    print("快取不存在，需要重新搜尋")

# 清除過期快取
count = cache.clear_expired()
print(f"清除 {count} 筆過期快取")

==============================================================================
"""

# ==============================================================================
# 引入必要的外部套件
# ==============================================================================

import hashlib  # 產生 MD5 hash，用於快速比對查詢
import json  # 將搜尋結果序列化為 JSON 字串
import sqlite3  # SQLite 資料庫操作（同步版本）
import time  # 時間處理（保留擴充性）
from datetime import datetime, timedelta  # 日期時間處理
from pathlib import Path  # 處理檔案路徑
from typing import Any, Optional  # 類型提示工具

# 引入設定模組
from .config import settings

# ==============================================================================
# 搜尋快取類別定義
# ==============================================================================


class SearchCache:
    """
    搜尋結果快取管理

    用途：
    ---------
    這個類別提供搜尋結果的快取功能，
    避免在短時間內重複呼叫 Brave Search API。

    快取機制：
    ---------
    1. 每次搜尋結果會儲存到 SQLite 資料庫
    2. 每筆快取都有過期時間（預設 24 小時）
    3. 超過過期時間的快取會在下次存取時被忽略
    4. 可以手動呼叫 clear_expired() 清除過期快取

    資料表結構：
    ---------
    search_cache 表格：
    - id：主鍵（自動遞增）
    - query：搜尋關鍵字
    - query_hash：關鍵字的 MD5 hash（唯一索引）
    - results：搜尋結果（JSON 格式字串）
    - created_at：建立時間
    - expires_at：過期時間

    使用範例：
    ---------
    from src.core.cache import SearchCache

    cache = SearchCache()

    # 儲存搜尋結果
    results = [
        {"title": "iPhone 15", "url": "https://example.com/iphone15"},
        {"title": "iPhone 15 評測", "url": "https://example.com/review"}
    ]
    cache.set("iphone 15", results)

    # 取得快取結果
    cached = cache.get("iphone 15")
    if cached:
        print(f"找到 {len(cached)} 筆快取結果")
    else:
        print("快取不存在")

    # 清除過期快取
    count = cache.clear_expired()
    print(f"清除 {count} 筆過期快取")
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化快取實例

        參數說明：
        ---------
        db_path (Optional[Path])：
          - 快取資料庫檔案路徑
          - 如果為 None，則使用設定中的預設資料庫路徑
          - 預設路徑：專案根目錄/data/shopping.db

        使用範例：
        ---------
        # 使用預設路徑
        cache = SearchCache()

        # 自訂路徑
        cache = SearchCache(db_path=Path("cache.db"))
        """

        # 設定資料庫檔案路徑
        # 如果有提供 db_path 就使用提供的，否則使用設定中的預設路徑
        self.db_path = db_path or settings.DATABASE_PATH

        # 初始化快取資料庫
        # 確保資料表已建立
        self._init_cache_db()

    # --------------------------------------------------------------------------
    # 快取資料庫初始化
    # ------------------------------------------------------

    def _init_cache_db(self) -> None:
        """
        初始化快取資料庫（內部方法）

        用途：
        ---------
        建立 search_cache 表格與必要索引。
        在初始化時自動呼叫，確保資料表存在。

        建立的表格：
        ---------
        search_cache：
        CREATE TABLE IF NOT EXISTS search_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            query_hash TEXT NOT NULL UNIQUE,
            results TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL
        )

        建立的索引：
        ---------
        1. idx_query_hash - 加速根據 hash 查找（唯一索引）
        2. idx_expires_at - 加速過期檢查

        返回值：
        ---------
        None
        """

        # 確保資料庫目錄存在
        # 如果資料夾不存在，自動建立
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 with 陳述句確保連線會自動關閉
        with sqlite3.connect(self.db_path) as conn:
            # 建立 search_cache 表格
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_hash TEXT NOT NULL UNIQUE,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)

            # 為 query_hash 建立唯一索引，確保不會有重複的關鍵字快取
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON search_cache(query_hash)")

            # 為 expires_at 建立索引，加速過期檢查
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON search_cache(expires_at)")

    # --------------------------------------------------------------------------
    # Hash 產生
    # ------------------------------------------------------

    def _generate_hash(self, query: str) -> str:
        """
        產生查詢的 MD5 hash（內部方法）

        用途：
        ---------
        將搜尋關鍵字轉換為 MD5 hash，
        用於作為快取的唯一識別碼。

        優點：
        ---------
        1. 長度固定：MD5 hash 總是 32 個字元
        2. 唯一性：相同的關鍵字會產生相同的 hash
        3. 效能：比直接使用長字串作為索引更快
        4. 安全性：不直接儲存原始關鍵字（簡易保護）

        參數說明：
        ---------
        query (str)：
          - 搜尋關鍵字
          - 例如："iPhone 15"

        返回值：
        ---------
        str：MD5 hash（32 個字元的十六進位字串）
          - 例如："a1b2c3d4e5f6789012345678abcdef12"

        使用範例：
        ---------
        hash = cache._generate_hash("iphone 15")
        print(hash)  # 輸出：97831e7e6e5f6789012345678abcdef12
        """

        # 將關鍵字編碼為 UTF-8 位元組
        # 然後計算 MD5 hash
        # 最後以十六進位字串形式返回
        return hashlib.md5(query.encode()).hexdigest()

    # --------------------------------------------------------------------------
    # 過期檢查
    # ------------------------------------------------------

    def _is_expired(self, created_at: datetime) -> bool:
        """
        檢查快取是否過期（內部方法）

        用途：
        ---------
        根據快取的建立時間與設定的過期時間，
        判斷快取是否已經過期。

        過期計算：
        ---------
        expires_at = created_at + CACHE_EXPIRY_HOURS（預設 24 小時）
        if now > expires_at:
            快取已過期
        else:
            快取仍然有效

        參數說明：
        ---------
        created_at (datetime)：
          - 快取的建立時間

        返回值：
        ---------
        bool：
          - True：快取已過期
          - False：快取仍然有效

        使用範例：
        ---------
        from datetime import datetime, timedelta

        # 快取建立於 12 小時前
        created_at = datetime.now() - timedelta(hours=12)
        cache._is_expired(created_at)  # False（仍然有效）

        # 快取建立於 26 小時前
        created_at = datetime.now() - timedelta(hours=26)
        cache._is_expired(created_at)  # True（已過期）
        """

        # 計算過期時間
        # created_at + 過期小時數 = expires_at
        expiry_time = created_at + timedelta(hours=settings.CACHE_EXPIRY_HOURS)

        # 比較當前時間與過期時間
        # now > expiry_time 表示已經過期
        return datetime.now() > expiry_time

    # --------------------------------------------------------------------------
    # 取得快取
    # ------------------------------------------------------

    def get(self, query: str) -> Optional[list[dict[str, Any]]]:
        """
        取得快取的搜尋結果

        用途：
        ---------
        從快取資料庫中取得指定關鍵字的搜尋結果，
        如果快取不存在或已過期，回傳 None。

        查詢流程：
        ---------
        1. 產生關鍵字的 MD5 hash
        2. 查找快取資料庫中是否有此 hash 的記錄
        3. 檢查快取是否過期（expires_at > now）
        4. 如果找到且未過期，反序列化 JSON 結果並回傳
        5. 否則回傳 None

        參數說明：
        ---------
        query (str)：
          - 搜尋關鍵字
          - 例如："iPhone 15"

        返回值：
        ---------
        Optional[list[dict[str, Any]]]：
          - list[dict[str, Any]]：快取的搜尋結果
            每個元素是字典，包含搜尋結果的一筆資料
          - None：快取不存在或已過期

        使用範例：
        ---------
        # 取得快取結果
        cached = cache.get("iphone 15")

        if cached:
            # 快取存在，使用快取結果
            print(f"找到 {len(cached)} 筆快取結果")
            for result in cached:
                print(f"標題：{result['title']}")
        else:
            # 快取不存在，需要重新搜尋
            print("快取不存在，需要重新搜尋")
        """

        # 產生關鍵字的 MD5 hash
        query_hash = self._generate_hash(query)

        # 使用 with 陳述句確保連線會自動關閉
        with sqlite3.connect(self.db_path) as conn:
            # 查找快取記錄
            # 1. 匹配 query_hash
            # 2. 檢查 expires_at > now（未過期）
            cursor = conn.execute(
                "SELECT results FROM search_cache WHERE query_hash = ? AND expires_at > ?",
                (
                    query_hash,                    # 匹配 hash
                    datetime.now().isoformat()     # 檢查是否過期
                )
            )

            # 取得一筆記錄
            row = cursor.fetchone()

            # 如果找到記錄
            if row:
                # 反序列化 JSON 字串為 Python 物件
                # row[0] 取出 results 欄位（JSON 字串）
                return json.loads(row[0])

        # 沒有找到快取或已過期
        return None

    # --------------------------------------------------------------------------
    # 儲存快取
    # ------------------------------------------------------

    def set(self, query: str, results: list[dict[str, Any]]) -> None:
        """
        儲存搜尋結果到快取

        用途：
        ---------
        將搜尋結果儲存到快取資料庫，
        以便未來相同的關鍵字可以直接使用快取。

        儲存流程：
        ---------
        1. 產生關鍵字的 MD5 hash
        2. 計算過期時間（now + CACHE_EXPIRY_HOURS）
        3. 將搜尋結果序列化為 JSON 字串
        4. 使用 INSERT OR REPLACE 儲存記錄
           - 如果 hash 已存在，則更新（REPLACE）
           - 如果 hash 不存在，則插入（INSERT）

        參數說明：
        ---------
        query (str)：
          - 搜尋關鍵字
          - 例如："iPhone 15"

        results (list[dict[str, Any]])：
          - 搜尋結果清單
          - 每個元素是字典，包含一筆搜尋結果

        使用範例：
        ---------
        results = [
            {
                "title": "iPhone 15",
                "url": "https://example.com/iphone15",
                "description": "最新款 iPhone 15"
            },
            {
                "title": "iPhone 15 評測",
                "url": "https://example.com/review",
                "description": "iPhone 15 詳細評測"
            }
        ]
        cache.set("iphone 15", results)
        """

        # 產生關鍵字的 MD5 hash
        query_hash = self._generate_hash(query)

        # 計算過期時間
        expires_at = datetime.now() + timedelta(hours=settings.CACHE_EXPIRY_HOURS)

        # 使用 with 陳述句確保連線會自動關閉
        with sqlite3.connect(self.db_path) as conn:
            # 執行 INSERT OR REPLACE 查詢
            # INSERT OR REPLACE 會在主鍵或唯一索引衝突時自動更新
            conn.execute(
                """
                INSERT OR REPLACE INTO search_cache (query, query_hash, results, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query,                          # query：原始關鍵字
                    query_hash,                     # query_hash：關鍵字 hash
                    json.dumps(results, ensure_ascii=False),  # results：JSON 序列化結果
                    datetime.now().isoformat(),     # created_at：建立時間
                    expires_at.isoformat()          # expires_at：過期時間
                )
            )

    # --------------------------------------------------------------------------
    # 清除過期快取
    # ------------------------------------------------------

    def clear_expired(self) -> int:
        """
        清除過期快取

        用途：
        ---------
        從快取資料庫中刪除所有已過期的記錄，
        回傳清除的筆數。

        查詢方式：
        ---------
        DELETE FROM search_cache WHERE expires_at < now

        返回值：
        ---------
        int：清除的記錄筆數
          - 0：沒有過期快取
          - > 0：清除的筆數

        使用範例：
        ---------
        # 清除過期快取
        count = cache.clear_expired()
        print(f"清除 {count} 筆過期快取")

        數據庫查詢範例：
        -----------------
        DELETE FROM search_cache WHERE expires_at < '2024-01-15T12:00:00'
        """

        # 使用 with 陳述句確保連線會自動關閉
        with sqlite3.connect(self.db_path) as conn:
            # 執行 DELETE 查詢
            # 刪除所有 expires_at < now 的記錄
            cursor = conn.execute(
                "DELETE FROM search_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )

            # 回傳清除的筆數
            return cursor.rowcount

    # --------------------------------------------------------------------------
    # 清理快取
    # ------------------------------------------------------

    def cleanup(self) -> None:
        """
        清理快取（顯示訊息版本）

        用途：
        ---------
        呼叫 clear_expired() 清除過期快取，
        並顯示清除的筆數訊息。

        訊息格式：
        ---------
        - 如果清除 0 筆：不顯示訊息
        - 如果清除 > 0 筆：顯示 "快取清理：已清除 X 筆過期資料"

        使用範例：
        ---------
        # 清理快取
        cache.cleanup()
        # 輸出：快取清理：已清除 5 筆過期資料

        定期清理建議：
        -------------
        - 建議在應用程式啟動時執行一次
        - 或設定定時任務定期執行（例如每小時）
        - 或在使用者要求時手動執行
        """

        # 清除過期快取
        count = self.clear_expired()

        # 如果有清除任何快取，顯示訊息
        if count > 0:
            print(f"快取清理：已清除 {count} 筆過期資料")
