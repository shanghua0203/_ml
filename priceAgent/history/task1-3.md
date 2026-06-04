# Role: 資深 Python AI 軟體架構師
# Task 3: 擴充進階 LLM 分析功能 (LLM Feature Enhancement)

請升級 LLM 判斷邏輯，強化比價與推薦品質：

1. **升級 Pydantic 模型**：
   - 檔案：`src/models/products.py`。
   - 動作：在 `PriceAnalysis` 中新增 `purchase_reason` (str, 購買理由與商品優勢) 與 `price_confidence` (int 0-100, 真實價格信心度)。

2. **強化價格萃取與驗證 (LLM Price Extraction)**：
   - 檔案：`src/agents/comparison_agent.py` (`_extract_price_with_llm`)。
   - 動作：修改 Prompt，要求 LLM 從爬蟲下來的 Markdown 中分辨「真實商品價格」與「配件/干擾價格」。強制 LLM 回傳 JSON 格式，包含：`real_price` 與 `confidence`。若信心度低於 50%，視為抓取失敗。

3. **強化風險判斷與購買理由**：
   - 檔案：`src/agents/price_guard.py` (`analyze_with_llm`)。
   - 動作：修改 Prompt，要求 LLM 不只判斷風險，還要根據 Markdown 內容生成「推薦購買理由」(purchase_reason)。並將此結果寫入 `PriceAnalysis` 模型中返回。