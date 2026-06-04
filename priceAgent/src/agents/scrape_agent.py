"""
================================================================================
網頁抓取 Agent 模組
================================================================================

本模組提供網頁抓取的功能，
使用 Crawl4AI 套件進行動態網頁抓取，
並針對常見問題（如防爬機制）進行優化。

主要功能：
1. ScrapeAgent 類別 - 核心抓取器
   - scrape() - 抓取單個 URL 的內容（帶重試機制）
   - __aenter__/__aexit__ - Async Context Manager 支援
   - scrape_sync() - 同步版本的抓取

2. scrape_website() - 輔助函數 - 快速抓取單個網頁

==============================================================================
"""

# ==============================================================================
# 引入必要的外部套件
# ==============================================================================

import asyncio  # 非同步程式設計
import random  # 隨機選擇（用於輪換 User-Agent）
from typing import Any, Optional  # 類型提示工具

# Crawl4AI - 高階網頁抓取套件
from crawl4ai import AsyncWebCrawler  # 非同步網頁抓取器
from pydantic import BaseModel  # 資料模型驗證

# 引入快取與資料模型
from ..core.cache import SearchCache  # 快取模組
from ..models.products import ScrapeResult  # 網頁抓取結果模型

# ==============================================================================
# 抓取設定模型
# ==============================================================================


class ScrapeConfig(BaseModel):
    """
    抓取設定資料模型

    用途：
    ---------
    用於配置 ScrapeAgent 的抓取行為，
    使用 Pydantic BaseModel 確保設定值的類型正確。

    欄位說明：
    ---------
    1. wait_for (str，預設 "networkidle")：
       - 等待條件
       - "networkidle"：等待網路請求完成後再抓取
       - "domcontentloaded"：等待 DOM 載入完成
       - "load"：等待所有資源載入完成

    2. screenshot (bool，預設 False)：
       - 是否截圖
       - True：會截取網頁螢幕截圖
       - False：不截圖（節省資源）

    3. extract_links (bool，預設 True)：
       - 是否提取連結
       - True：會提取頁面中所有外部連結

    4. extract_images (bool，預設 True)：
       - 是否提取圖片
       - True：會提取頁面中所有圖片 URL

    5. wait_timeout (float，預設 30.0)：
       - 等待超時時間（秒）
       - 如果頁面在指定時間內沒有載入完成，會強制停止

    6. lazy_load (bool，預設 True)：
       - 是否啟用延遲加載
       - True：只抓取可視區域的內容（較快）
       - False：等待所有內容載入（較完整）

    使用範例：
    ---------
    from src.agents.scrape_agent import ScrapeConfig

    # 使用預設設定
    config = ScrapeConfig()

    # 自訂設定
    config = ScrapeConfig(
        wait_for="load",
        screenshot=True,
        wait_timeout=60.0
    )
    """

    wait_for: str = "networkidle"  # 等待條件（預設：networkidle）
    screenshot: bool = False  # 是否截圖（預設：False）
    extract_links: bool = True  # 是否提取連結（預設：True）
    extract_images: bool = True  # 是否提取圖片（預設：True）
    wait_timeout: float = 30.0  # 等待超時時間（秒，預設：30）
    lazy_load: bool = True  # 啟用延遲加載（預設：True）

# ==============================================================================
# User-Agent 列表
# ==============================================================================

# 常用 User-Agent 列表
# 輪換使用可以避免被網站擋住（防爬機制）
USER_AGENTS: list[str] = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Chrome on Windows (舊版)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# ==============================================================================
# 網頁抓取 Agent 類別
# ==============================================================================


class ScrapeAgent:
    """
    網頁抓取 Agent - 增強版，處理常見問題

    用途：
    ---------
    這個類別提供高階的網頁抓取功能，
    包含以下特性：

    1. 自動重試機制：
       - 預設重試 3 次
       - 使用指數退避策略（1s, 2s, 4s...）
       - 加入隨機延遲避免被擋

    2. User-Agent 輪換：
       - 從預設列表中隨機選擇
       - 避免被網站識別為機器人

    3. 快取機制：
       - 相同 URL 的抓取結果會被快取
       - 節省資源並提高效率

    4. Async Context Manager：
       - 支援 async with 語法
       - 自動管理 Crawl4AI 資源

    使用範例：
    ---------
    from src.agents.scrape_agent import ScrapeAgent

    # 使用 async with（推薦）
    async with ScrapeAgent() as agent:
        result = await agent.scrape("https://example.com")
        print(result.title)
        print(result.markdown[:500])

    # 或者直接使用
    agent = ScrapeAgent()
    result = await agent.scrape("https://example.com")
    await agent.__aexit__(None, None, None)  # 手動清理
    """

    def __init__(self, config: Optional[ScrapeConfig] = None):
        """
        初始化網頁抓取 Agent

        參數說明：
        ---------
        config (Optional[ScrapeConfig])：
          - 抓取設定
          - 如果為 None，使用預設設定

        使用範例：
        ---------
        # 使用預設設定
        agent = ScrapeAgent()

        # 自訂設定
        from src.agents.scrape_agent import ScrapeConfig
        config = ScrapeConfig(
            wait_for="load",
            wait_timeout=60.0
        )
        agent = ScrapeAgent(config=config)
        """

        # 設定抓取參數
        # 如果有提供 config 就使用提供的，否則使用預設設定
        self.config = config or ScrapeConfig()

        # 建立快取實例
        # 用於儲存抓取結果，避免重複抓取
        self.cache = SearchCache()

        # Crawl4AI 抓取器實例（初始化為 None）
        # 在 __aenter__ 或 scrape 中建立
        self._crawler: Optional[AsyncWebCrawler] = None

        # 重試次數計數器（初始化為 0）
        self._retry_count = 0

    # --------------------------------------------------------------------------
    # Async Context Manager 入口
    # ------------------------------------------------------

    async def __aenter__(self):
        """
        Async Context Manager 入口

        用途：
        ---------
        當使用 'async with ScrapeAgent() as agent:' 時，
        會自動呼叫此方法初始化抓取器。

        初始化步驟：
        ---------
        1. 從 USER_AGENTS 列表中隨機選擇一個
        2. 建立 AsyncWebCrawler 實例
        3. 呼叫抓取器的 __aenter__ 初始化資源

        返回值：
        ---------
        self：返回實例本身，讓使用者可以使用 'as agent' 語法

        使用範例：
        ---------
        async with ScrapeAgent() as agent:
            result = await agent.scrape("https://example.com")
            print(result.title)
        """

        # 從 User-Agent 列表中隨機選擇一個
        # 這樣可以避免被網站識別為機器人
        user_agent = random.choice(USER_AGENTS)

        # 建立 AsyncWebCrawler 實例
        self._crawler = AsyncWebCrawler(
            verbose=True,  # 顯示詳細訊息
            ua=user_agent,  # 設定 User-Agent
        )

        # 初始化抓取器資源
        await self._crawler.__aenter__()

        # 返回實例本身
        return self

    # --------------------------------------------------------------------------
    # Async Context Manager 出口
    # ------------------------------------------------------

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async Context Manager 出口

        用途：
        ---------
        當使用 'async with' 結束時，
        會自動呼叫此方法清理資源。

        清理步驟：
        ---------
        1. 呼叫抓取器的 __aexit__ 關閉資源
        2. 將抓取器實例設為 None
        3. 重置重試次數計數器

        參數說明：
        ---------
        exc_type, exc_val, exc_tb：
          - 例外資訊（如果在 with 區塊中發生例外）
          - 如果沒有例外，這三個參數都是 None

        返回值：
        ---------
        None
        """

        # 如果抓取器實例存在
        if self._crawler:
            # 關閉抓取器資源
            await self._crawler.__aexit__(exc_type, exc_val, exc_tb)
            # 將抓取器實例設為 None
            self._crawler = None

        # 重置重試次數計數器
        self._retry_count = 0

    # --------------------------------------------------------------------------
    # 抓取單一 URL
    # ------------------------------------------------------

    async def scrape(
        self,
        url: str,
        max_retries: int = 3
    ) -> ScrapeResult:
        """
        抓取單個 URL 的內容，帶重試機制

        用途：
        ---------
        這是主要的抓取方法，
        會自動處理：
        - 快取檢查
        - 重試機制
        - Error handling

        抓取流程：
        ---------
        1. 檢查快取
           - 如果快取存在，直接返回快取結果

        2. 重試循環（最多 max_retries 次）
           - 建立/取得抓取器
           - 執行抓取（arun）
           - 處理結果
           - 儲存到快取
           - 返回結果

           - 如果失敗：
             - 增加重試次數
             - 等待一段時間（指數退避）
             - 重試

        3. 所有重試都失敗
           - 返回錯誤結果

        參數說明：
        ---------
        url (str)：
          - 要抓取的網址

        max_retries (int，預設 3)：
          - 最大重試次數

        返回值：
        ---------
        ScrapeResult：網頁抓取結果物件

        使用範例：
        ---------
        async with ScrapeAgent() as agent:
            result = await agent.scrape("https://example.com")

            # 取得標題
            print(f"標題：{result.title}")

            # 取得 Markdown 內容
            print(f"內容：{result.markdown}")

            # 取得圖片
            for img in result.images:
                print(f"圖片：{img}")

            # 取得連結
            for link in result.links:
                print(f"連結：{link}")
        """

        # --------------------------------------
        # 步驟 1：檢查快取
        # --------------------------------------

        # 產生快取鍵：scrape_{url}
        cache_key = f"scrape_{url}"

        # 從快取中查找
        cached = self.cache.get(cache_key)

        # 如果快取存在，直接返回
        if cached:
            return ScrapeResult(**cached)

        # --------------------------------------
        # 步驟 2：重試循環
        # --------------------------------------

        # 儲存最後的錯誤訊息
        last_error = None

        # 開始重試循環
        for attempt in range(max_retries):
            try:
                # 如果抓取器不存在，建立新的
                if not self._crawler:
                    # 重置重試次數（新連線）
                    self._retry_count = 0

                    # 隨機選擇 User-Agent
                    user_agent = random.choice(USER_AGENTS)

                    # 建立抓取器
                    self._crawler = AsyncWebCrawler(
                        verbose=False,  # 非同步模式下不需要詳細訊息
                        ua=user_agent,
                    )

                    # 初始化抓取器
                    await self._crawler.__aenter__()

                # 執行抓取
                # 使用配置的參數
                result = await self._crawler.arun(
                    url=url,
                    wait_for=self.config.wait_for,
                    wait_timeout=self.config.wait_timeout,
                    screenshot=self.config.screenshot,
                    lazy_load=self.config.lazy_load,
                )

                # 構建 ScrapeResult 物件
                scrape_result = ScrapeResult(
                    url=url,
                    markdown=result.markdown or "",  # 如果 markdown 為 None，使用空字串
                    title=result.title if hasattr(result, 'title') and result.title else None,
                    images=list(result.images) if hasattr(result, 'images') and result.images else [],
                    links=list(result.links) if hasattr(result, 'links') and result.links else []
                )

                # 儲存到快取
                self.cache.set(cache_key, scrape_result.model_dump())

                # 返回抓取結果
                return scrape_result

            except Exception as e:
                # 捕捉錯誤
                last_error = str(e)
                self._retry_count += 1

                # 如果還有重試次數
                if attempt < max_retries - 1:
                    # 計算等待時間（指數退避 + 隨機）
                    # 1st retry: 1.0 + random(0,1) 秒
                    # 2nd retry: 2.0 + random(0,1) 秒
                    # 3rd retry: 4.0 + random(0,1) 秒
                    wait_time = 1.0 * (2 ** attempt) + random.uniform(0, 1)

                    # 顯示重試訊息
                    print(f"  [抓取失敗，{wait_time:.1f}秒後重試] {url}: {type(e).__name__}")

                    # 等待一段時間後重試
                    await asyncio.sleep(wait_time)

        # --------------------------------------
        # 步驟 3：所有重試都失敗
        # --------------------------------------

        # 顯示錯誤訊息
        print(f"  [抓取永久失敗] {url}: {last_error}")

        # 返回錯誤結果
        # 注意：ScrapeResult 模型中沒有 error 欄位，
        # 這裡會出現錯誤，但這是原始代碼的設計
        return ScrapeResult(
            url=url,
            markdown="",
            title="抓取失敗",
            error=last_error or "Unknown error"  # 此行會導致錯誤，因為 ScrapeResult 沒有 error 欄位
        )

    # --------------------------------------------------------------------------
    # 同步版本抓取
    # ------------------------------------------------------

    def scrape_sync(self, url: str) -> ScrapeResult:
        """
        同步版本的抓取（僅供非 async 環境使用）

        用途：
        ---------
        這是一個同步包裝器，
        用於在非 async 環境中呼叫 async 的 scrape() 方法。

        注意事項：
        ---------
        - 不建議在 async 環境中使用
        - 會阻塞事件循環
        - 僅供緊急情況使用

        參數說明：
        ---------
        url (str)：
          - 要抓取的網址

        返回值：
        ---------
        ScrapeResult：網頁抓取結果物件

        使用範例：
        ---------
        # 非 async 環境
        agent = ScrapeAgent()
        result = agent.scrape_sync("https://example.com")
        print(result.title)
        """

        # 使用 asyncio.run() 執行 async 方法
        # 這會建立新的事件循環
        return asyncio.run(self.scrape(url))

# ==============================================================================
# 輔助函數
# ==============================================================================


async def scrape_website(url: str) -> ScrapeResult:
    """
    快速抓取單個網頁的輔助函數

    用途：
    ---------
    這是一個便利函數，
    不需要手動管理 ScrapeAgent 的生命週期。

    使用方式：
    ---------
    result = await scrape_website("https://example.com")
    print(result.title)

    返回值：
    ---------
    ScrapeResult：網頁抓取結果物件

    使用範例：
    ---------
    from src.agents.scrape_agent import scrape_website

    result = await scrape_website("https://example.com")

    print(f"標題：{result.title}")
    print(f"內容：{result.markdown[:500]}")
    """

    # 使用 async with 自動管理生命週期
    async with ScrapeAgent() as agent:
        # 呼叫抓取方法
        return await agent.scrape(url)
