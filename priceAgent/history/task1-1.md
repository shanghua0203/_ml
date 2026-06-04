# Role: 資深 Python AI 軟體架構師
# Task 1: 核心 Bug 修復 (Critical Bug Fixes)

請嚴格執行以下程式碼修復，確保系統穩定性，完成後請回報修改了哪些檔案：

1. **修復 LLM JSON 解析錯誤**：
   - 檔案：`src/agents/price_guard.py` 的 `_parse_llm_response`。
   - 動作：在 `json.loads` 前，使用 Regex (`re.sub`) 強制清除字串前後可能存在的 Markdown 標籤（如 ```json 與 ```），避免 DecodeError。

2. **修復無效的 Pydantic 驗證器**：
   - 檔案：`src/models/products.py` 的 `FraudDetectionResult`。
   - 動作：實作 `@field_validator("suspicious_keywords")`。邏輯：若傳入的 `v` (list) 長度大於 0，需連帶確保物件的 `has_suspicious_text` 被設為 True（可利用 `@model_validator` 或直接在處理邏輯中卡控）。

3. **修復資源生命週期 (Memory Leak)**：
   - 檔案：`src/agents/comparison_agent.py` 與 `src/main.py`。
   - 動作：將 `ComparisonAgent` 實作為 Async Context Manager (`__aenter__`, `__aexit__`)，在內部正確管理 `self.scrape_agent` 的開關。修改 `main.py` 的呼叫方式，改用 `async with ComparisonAgent() as agent:`。