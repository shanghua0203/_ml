# Role: 資深 Python AI 軟體架構師
# Task 2: 非同步效能優化 (AsyncIO & DB Optimization)

請嚴格執行以下效能優化，這會牽動底層架構，請小心處理：

1. **全面替換為非同步資料庫 (aiosqlite)**：
   - 檔案：`requirements.txt`, `src/utils/db.py`, `src/core/cache.py`, `src/main.py`。
   - 動作：安裝並引入 `aiosqlite` 取代 `sqlite3`。將所有資料庫 CRUD 方法改為 `async def`，內部使用 `await aiosqlite.connect()`。
   - 注意：同步更新呼叫端（如 `main.py` 中的 `db.get_history` 等）加上 `await`，確保事件迴圈 (Event Loop) 不被 I/O 阻塞。

2. **加入爬蟲併發控制 (Concurrency Limiting)**：
   - 檔案：`src/agents/comparison_agent.py`。
   - 動作：在 `compare` 方法中，使用 `asyncio.Semaphore(3)` 來限制 `_scrape_and_analyze` 的併發數量。避免 `asyncio.gather` 瞬間啟動過多無頭瀏覽器導致 OOM (Out of Memory)。