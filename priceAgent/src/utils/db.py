"""
================================================================================
資料庫操作工具模組
================================================================================

本模組提供非同步的 SQLite 資料庫操作介面，
用於儲存比價結果與搜尋日誌。

主要功能：
1. Database 類別 - 管理 shopping_history 和 search_logs 表格
   - save_shopping_result() - 儲存比價結果
   - save_search_log() - 儲存搜尋日誌
   - get_history() - 取得歷史比價紀錄
   - get_recent_searches() - 取得最近搜尋關鍵字
   - get_stats() - 取得統計資料

使用方式：
---------
from src.utils.db import db

# 儲存比價結果
result_id = await db.save_shopping_result(
    query="iphone 15",
    product_name="iPhone 15",
    product_url="https://momo.tw/iphone15",
    source_platform="momo",
    price=32900,
    risk_level="low",
    is_recommendation=True
)

# 取得歷史紀錄
history = await db.get_history(limit=10)

# 取得統計資料
stats = await db.get_stats()

==============================================================================
"""

# ==============================================================================
# 引入必要的外部套件
# ==============================================================================

import asyncio  # 非同步程式設計（保留擴充性）
from datetime import datetime  # 日期時間處理
from pathlib import Path  # 處理檔案路徑
from typing import Any, Optional  # 類型提示工具

import aiosqlite  # 非同步 SQLite 資料庫操作套件

# 引入設定模組
from ..core.config import settings

# ==============================================================================
# 資料庫類別定義
# ==============================================================================


class Database:
    """
    SQLite 資料庫管理類別（非同步版本）

    用途：
    ---------
    這個類別提供對 SQLite 資料庫的非同步操作介面，
    用於儲存與查詢比價機器人的歷史紀錄。

    資料表結構：
    ---------
    1. shopping_history（比價歷史表）：
       - id：主鍵（自動遞增）
       - search_query：搜尋關鍵字
       - product_name：商品名稱
       - product_url：商品網址
       - source_platform：來源平台（momo/shopee/pchome/apple）
       - price：商品價格
       - risk_level：風險等級（low/medium/high）
       - is_recommendation：是否為推薦商品（1=是，0=否）
       - scraped_at：抓取時間
       - created_at：建立時間（自動設定）

    2. search_logs（搜尋日誌表）：
       - id：主鍵（自動遞增）
       - query：搜尋關鍵字
       - results_count：搜尋結果數量
       - executed_at：執行時間（自動設定）

    使用範例：
    ---------
    from src.utils.db import Database
    from pathlib import Path

    # 建立資料庫實例
    db = Database(db_path=Path("data/shopping.db"))

    # 儲存比價結果
    result_id = await db.save_shopping_result(
        query="iphone 15",
        product_name="iPhone 15",
        product_url="https://example.com",
        source_platform="momo",
        price=32900,
        risk_level="low"
    )
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化資料庫實例

        參數說明：
        ---------
        db_path (Optional[Path])：
          - 資料庫檔案路徑
          - 如果為 None，則使用設定中的預設路徑
          - 預設路徑：專案根目錄/data/shopping.db

        使用範例：
        ---------
        # 使用預設路徑
        db = Database()

        # 自訂路徑
        db = Database(db_path=Path("custom_path.db"))
        """

        # 設定資料庫檔案路徑
        # 如果有提供 db_path 就使用提供的，否則使用設定中的預設路徑
        self.db_path = db_path or settings.DATABASE_PATH

    # --------------------------------------------------------------------------
    # 資料庫初始化
    # ----------------------------------------------------------

    async def _init_db(self, conn: aiosqlite.Connection) -> None:
        """
        初始化資料庫與表格（內部方法）

        用途：
        ---------
        在每次資料庫連線後呼叫，
        確保所需的資料表與索引都已建立。

        建立的表格：
        ---------
        1. shopping_history（比價歷史表）：
           CREATE TABLE IF NOT EXISTS shopping_history (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               search_query TEXT NOT NULL,
               product_name TEXT NOT NULL,
               product_url TEXT NOT NULL,
               source_platform TEXT NOT NULL,
               price REAL NOT NULL,
               risk_level TEXT NOT NULL,
               is_recommendation INTEGER NOT NULL,
               scraped_at TIMESTAMP NOT NULL,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )

        2. search_logs（搜尋日誌表）：
           CREATE TABLE IF NOT EXISTS search_logs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               query TEXT NOT NULL,
               results_count INTEGER NOT NULL,
               executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )

        建立的索引：
        ---------
        1. idx_search_query - 加速根據搜尋關鍵字查詢
        2. idx_scraped_at - 加速根據抓取時間排序
        3. idx_query - 加速根據查詢關鍵字查詢

        參數說明：
        ---------
        conn (aiosqlite.Connection)：
          - 已建立的資料庫連線物件

        返回值：
        ---------
        None
        """

        # 建立 shopping_history 表格
        # 用於儲存每次比價的結果
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS shopping_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_query TEXT NOT NULL,
                product_name TEXT NOT NULL,
                product_url TEXT NOT NULL,
                source_platform TEXT NOT NULL,
                price REAL NOT NULL,
                risk_level TEXT NOT NULL,
                is_recommendation INTEGER NOT NULL,
                scraped_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 為 search_query 建立索引，加速搜尋
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_search_query ON shopping_history(search_query)")

        # 為 scraped_at 建立索引，加速時間排序
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_scraped_at ON shopping_history(scraped_at)")

        # 建立 search_logs 表格
        # 用於記錄每次搜尋的行為
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                results_count INTEGER NOT NULL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 為 query 建立索引，加速搜尋
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON search_logs(query)")

        # 提交變更
        # 在 aiosqlite 中，變更必須明確提交才能永久保存
        await conn.commit()

    async def _get_connection(self) -> aiosqlite.Connection:
        """
        取得資料庫連線（內部方法）

        用途：
        ---------
        這個方法會建立新的資料庫連線，
        並自動呼叫 _init_db 來初始化資料表。

        返回值：
        ---------
        aiosqlite.Connection：已初始化的資料庫連線物件

        使用範例：
        ---------
        conn = await self._get_connection()
        try:
            # 使用連線進行操作
            cursor = await conn.execute("SELECT * FROM shopping_history")
            rows = await cursor.fetchall()
        finally:
            await conn.close()
        """

        # 確保資料庫目錄存在
        # 如果資料夾不存在，自動建立
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 建立新的資料庫連線
        conn = await aiosqlite.connect(self.db_path)

        # 初始化資料表
        await self._init_db(conn)

        # 返回連線物件
        return conn

    # --------------------------------------------------------------------------
    # 儲存操作
    # ----------------------------------------------------------

    async def save_shopping_result(
        self,
        query: str,
        product_name: str,
        product_url: str,
        source_platform: str,
        price: float,
        risk_level: str,
        is_recommendation: bool = False
    ) -> int:
        """
        儲存比價結果到資料庫

        用途：
        ---------
        將一次比價的結果儲存到 shopping_history 表格，
        供後續查詢與統計使用。

        參數說明：
        ---------
        query (str)：
          - 搜尋關鍵字
          - 例如："iPhone 15"

        product_name (str)：
          - 商品名稱
          - 例如："iPhone 15 128GB 黑色"

        product_url (str)：
          - 商品網址
          - 例如："https://momo.tw/iphone15"

        source_platform (str)：
          - 來源平台
          - 可能值："momo"、"shopee"、"pchome"、"apple" 等

        price (float)：
          - 商品價格（新台幣）
          - 例如：32900.0

        risk_level (str)：
          - 風險等級
          - 可能值："low"（低風險）、"medium"（中風險）、"high"（高風險）

        is_recommendation (bool，預設 False)：
          - 是否為推薦商品
          - True：會在比價結果中被標記為推薦
          - False：一般商品

        返回值：
        ---------
        int：儲存的記錄 ID（last_insert_rowid）
          - > 0：成功，返回記錄 ID
          - 0：失敗

        使用範例：
        ---------
        result_id = await db.save_shopping_result(
            query="iphone 15",
            product_name="iPhone 15",
            product_url="https://momo.tw/iphone15",
            source_platform="momo",
            price=32900,
            risk_level="low",
            is_recommendation=True
        )
        print(f"儲存成功，ID：{result_id}")

        數據庫儲存範例：
        -----------------
        INSERT INTO shopping_history
        (search_query, product_name, product_url, source_platform, price, risk_level, is_recommendation, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        參數：("iphone 15", "iPhone 15", "https://momo.tw/iphone15", "momo", 32900, "low", 1, "2024-01-15T10:30:00")
        """

        # 取得資料庫連線
        # 使用 async with 確保連線會自動關閉
        async with aiosqlite.connect(self.db_path) as conn:
            # 初始化資料表（如果還沒有）
            await self._init_db(conn)

            # 執行 INSERT 查詢
            # 使用參數化查詢防止 SQL Injection 攻擊
            await conn.execute(
                """
                INSERT INTO shopping_history
                (search_query, product_name, product_url, source_platform, price, risk_level, is_recommendation, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query,           # search_query：搜尋關鍵字
                    product_name,    # product_name：商品名稱
                    product_url,     # product_url：商品網址
                    source_platform, # source_platform：來源平台
                    price,           # price：商品價格
                    risk_level,      # risk_level：風險等級
                    1 if is_recommendation else 0,  # is_recommendation：是否推薦（1 或 0）
                    datetime.now().isoformat()      # scraped_at：抓取時間（ISO 格式）
                )
            )

            # 提交變更
            await conn.commit()

            # 取得最後插入的記錄 ID
            cursor = await conn.execute("SELECT last_insert_rowid()")
            result = await cursor.fetchone()

            # 返回記錄 ID，如果失敗則返回 0
            return result[0] if result else 0

    async def save_search_log(self, query: str, results_count: int) -> int:
        """
        儲存搜尋日誌到資料庫

        用途：
        ---------
        記錄每次搜尋的行為，包括關鍵字與結果數量，
        用於統計與分析搜尋趨勢。

        參數說明：
        ---------
        query (str)：
          - 搜尋關鍵字
          - 例如："MacBook Pro"

        results_count (int)：
          - 搜尋結果數量
          - 例如：5

        返回值：
        ---------
        int：儲存的記錄 ID（last_insert_rowid）
          - > 0：成功，返回記錄 ID
          - 0：失敗

        使用範例：
        ---------
        result_id = await db.save_search_log(
            query="macbook pro",
            results_count=5
        )

        數據庫儲存範例：
        -----------------
        INSERT INTO search_logs (query, results_count) VALUES (?, ?)
        參數：("macbook pro", 5)
        """

        # 取得資料庫連線
        async with aiosqlite.connect(self.db_path) as conn:
            # 初始化資料表（如果還沒有）
            await self._init_db(conn)

            # 執行 INSERT 查詢
            await conn.execute(
                "INSERT INTO search_logs (query, results_count) VALUES (?, ?)",
                (
                    query,           # query：搜尋關鍵字
                    results_count    # results_count：搜尋結果數量
                )
            )

            # 提交變更
            await conn.commit()

            # 取得最後插入的記錄 ID
            cursor = await conn.execute("SELECT last_insert_rowid()")
            result = await cursor.fetchone()

            # 返回記錄 ID，如果失敗則返回 0
            return result[0] if result else 0

    # --------------------------------------------------------------------------
    # 查詢操作
    # ----------------------------------------------------------

    async def get_history(
        self,
        query: Optional[str] = None,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        取得歷史比價紀錄

        用途：
        ---------
        查詢 shopping_history 表格，取得比價歷史紀錄。
        可以篩選特定關鍵字，並限制回傳數量。

        查詢選項：
        ---------
        1. 不指定 query：回傳所有比價紀錄
        2. 指定 query：回傳符合關鍵字的比價紀錄（LIKE 模糊搜尋）

        排序方式：
        ---------
        - 按 created_at 降序（DESC）排序
        - 最新的比價結果會顯示在最前面

        分頁方式：
        ---------
        - 使用 LIMIT 子句限制回傳數量
        - 預設回傳最近的 10 筆紀錄

        參數說明：
        ---------
        query (Optional[str])：
          - 搜尋關鍵字（可選）
          - 如果提供，會使用 LIKE 模糊搜尋
          - 例如："iphone" 會匹配 "iPhone 15"、"iPhone 14" 等

        limit (int，預設 10)：
          - 最多回傳的紀錄數量

        返回值：
        ---------
        list[dict[str, Any]]：比價紀錄清單
          - 每個元素是字典，包含一筆比價結果
          - 空列表表示沒有符合條件的紀錄

        每筆紀錄的欄位：
        -----------------
        {
            "id": 1,
            "search_query": "iphone 15",
            "product_name": "iPhone 15",
            "product_url": "https://example.com/iphone15",
            "source_platform": "momo",
            "price": 32900.0,
            "risk_level": "low",
            "is_recommendation": 1,
            "scraped_at": "2024-01-15T10:30:00",
            "created_at": "2024-01-15T10:30:00"
        }

        使用範例：
        ---------
        # 取得最近 10 筆比價紀錄
        history = await db.get_history(limit=10)

        # 取得符合關鍵字的比價紀錄
        history = await db.get_history(query="iphone 15", limit=10)

        # 顯示歷史紀錄
        for record in history:
            print(f"📅 {record['created_at']}")
            print(f"   搜尋：{record['search_query']}")
            print(f"   商品：{record['product_name']}")
            print(f"   價格：NT${record['price']:,.0f}")
            print(f"   平台：{record['source_platform']}")
            print()
        """

        # 取得資料庫連線
        async with aiosqlite.connect(self.db_path) as conn:
            # 初始化資料表（如果還沒有）
            await self._init_db(conn)

            # 設定 Row Factory，讓結果以字典形式返回
            # aiosqlite.Row 允許我們用欄位名稱存取資料
            conn.row_factory = aiosqlite.Row

            # 判斷是否需要篩選關鍵字
            if query:
                # 有指定關鍵字：使用 LIKE 模糊搜尋
                cursor = await conn.execute(
                    """
                    SELECT * FROM shopping_history
                    WHERE search_query LIKE ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (
                        f"%{query}%",  # LIKE 模式：關鍵字前後加 % 進行模糊匹配
                        limit          # 限制回傳數量
                    )
                )
            else:
                # 沒指定關鍵字：回傳所有紀錄
                cursor = await conn.execute(
                    "SELECT * FROM shopping_history ORDER BY created_at DESC LIMIT ?",
                    (limit,)  # 只傳入 limit 參數
                )

            # 取得所有符合條件的記錄
            rows = await cursor.fetchall()

            # 將 Row 物件轉換為字典
            # [dict(row) for row in rows] 會將每一筆 Row 轉換為字典
            return [dict(row) for row in rows]

    async def get_recent_searches(self, limit: int = 5) -> list[str]:
        """
        取得最近搜尋關鍵字

        用途：
        ---------
        從 search_logs 表格中取得最近的搜尋關鍵字，
        用於顯示使用者的搜尋歷史。

        查詢方式：
        ---------
        - 使用 DISTINCT 去除重複的關鍵字
        - 按 executed_at 降序排序（最新的在前）
        - 使用 LIMIT 限制回傳數量

        參數說明：
        ---------
        limit (int，預設 5)：
          - 最多回傳的關鍵字數量

        返回值：
        ---------
        list[str]：搜尋關鍵字清單
          - 按時間倒序排列（最新的在前）
          - 空列表表示沒有搜尋紀錄

        使用範例：
        ---------
        # 取得最近 5 筆搜尋關鍵字
        searches = await db.get_recent_searches(limit=5)
        print(searches)
        # 輸出：['macbook pro', 'iphone 15', 'airpods', 'ipad', 'apple watch']

        數據庫查詢範例：
        -----------------
        SELECT DISTINCT query FROM search_logs
        ORDER BY executed_at DESC LIMIT 5
        """

        # 取得資料庫連線
        async with aiosqlite.connect(self.db_path) as conn:
            # 初始化資料表（如果還沒有）
            await self._init_db(conn)

            # 執行查詢
            cursor = await conn.execute(
                "SELECT DISTINCT query FROM search_logs ORDER BY executed_at DESC LIMIT ?",
                (limit,)  # 限制回傳數量
            )

            # 取得所有符合條件的記錄
            rows = await cursor.fetchall()

            # 提取關鍵字（每筆記錄只有一個欄位：query）
            # row[0] 取出第一個欄位（query 欄位）
            return [row[0] for row in rows]

    # --------------------------------------------------------------------------
    # 統計操作
    # ----------------------------------------------------------

    async def get_stats(self) -> dict[str, int]:
        """
        取得統計資料

        用途：
        ---------
        統計比價結果與搜尋日誌的總數量，
        用於顯示應用程式的使用情況。

        統計項目：
        ---------
        1. total_shopping_records：
           - shopping_history 表格的總筆數
           - 代表總共進行了多少次比價

        2. total_searches：
           - search_logs 表格的總筆數
           - 代表總共進行了多少次搜尋

        返回值：
        ---------
        dict[str, int]：統計資料字典
          - {"total_shopping_records": 100, "total_searches": 500}
          - 兩個鍵都是整數，表示記錄筆數

        使用範例：
        ---------
        stats = await db.get_stats()
        print(f"總比價紀錄：{stats['total_shopping_records']:,}")
        print(f"總搜尋次數：{stats['total_searches']:,}")

        數據庫查詢範例：
        -----------------
        1. SELECT COUNT(*) FROM shopping_history
           - 回傳 shopping_history 表格的總筆數

        2. SELECT COUNT(*) FROM search_logs
           - 回傳 search_logs 表格的總筆數
        """

        # 取得資料庫連線
        async with aiosqlite.connect(self.db_path) as conn:
            # 初始化資料表（如果還沒有）
            await self._init_db(conn)

            # 查詢 shopping_history 總筆數
            cursor = await conn.execute("SELECT COUNT(*) FROM shopping_history")
            total_shopping = (await cursor.fetchone())[0]

            # 查詢 search_logs 總筆數
            cursor = await conn.execute("SELECT COUNT(*) FROM search_logs")
            total_searches = (await cursor.fetchone())[0]

            # 返回統計資料
            return {
                "total_shopping_records": total_shopping,  # 比價紀錄總數
                "total_searches": total_searches           # 搜尋次數總數
            }


# ==============================================================================
# 全域資料庫實例
# ==============================================================================

# 建立 Database 類別的全域實例
# 這是整個專案唯一的資料庫實例
# 所有模組都可以透過 'from src.utils.db import db' 來存取資料庫
db = Database()
