# 優化專案中 LLM 的 JSON 解析邏輯，將原本依賴 Regex 和 manual JSON parsing 的脆弱作法，全面升級為 LangChain 的 `with_structured_output` 結合 Pydantic 模型，以確保 LLM 輸出的穩定性。

## 請嚴格遵循以下步驟執行：

### 【第一階段：安全備份】
1. 在開始任何修改前，請先執行 `git add .` 與 `git commit -m "chore: backup before refactoring LLM json parsing"`。這是為了確保若後續修改失敗，我們有乾淨的還原點。

### 【第二階段：重構 comparison_agent.py】
1. 開啟 `src/agents/comparison_agent.py`。
2. 引入 `pydantic` 的 `BaseModel` 與 `Field`。
3. 針對 `_extract_price_with_llm` 方法：
   - 建立一個 Pydantic 模型（例如 `PriceExtraction`），包含 `real_price` (float) 與 `confidence` (int)。
   - 將原本的 `self.llm.ainvoke(prompt)` 改為 `self.llm.with_structured_output(PriceExtraction).ainvoke(prompt)`。
   - 移除所有的 `re.sub`、`json.loads` 以及對 `cleaned` 字串的 Regex 解析，直接使用回傳的 Pydantic 物件屬性。
4. 針對 `_rank_results_with_llm` 方法：
   - 建立一個 Pydantic 模型（例如 `RankedURLs`），包含 `ranked_urls` (list[str])。
   - 使用 `with_structured_output(RankedURLs)` 來重構 invocation。
   - 移除所有手動的 JSON 解析與 Regex 程式碼。
5. 更新對應的 prompt，可以簡化原本「強制輸出 JSON 格式」的複雜文字，因為 LangChain 會自動處理 schema。

### 【第三階段：重構 price_guard.py】
1. 開啟 `src/agents/price_guard.py`。
2. 找出裡面所有呼叫 LLM 並要求回傳 JSON 格式的地方（例如 `analyze_with_llm` 等方法）。
3. 建立對應的 Pydantic 模型。
4. 套用 `with_structured_output()`。
5. 移除 Regex 和 `json.loads` 相關的防呆程式碼，依賴 Pydantic 來確保格式正確。

### 【第四階段：測試與驗證】
1. 修改完成後，請執行專案的測試（例如 `python test_scrape.py` 或是執行主程式 `python src/main.py` 帶入簡單的搜尋關鍵字）來驗證程式是否正常運作。
2. 檢查是否有 Pydantic 解析錯誤或 Ollama 支援度的問題。

### 【第五階段：還原或完成】
1. 如果測試過程中發現嚴重錯誤且嘗試修復 2 次仍無法解決（例如目前的 Ollama 模型不支援 structured output 導致不斷 crash），請執行 `git reset --hard HEAD~1` 退回到原本的版本，並向我報告遇到了什麼困難。
2. 如果測試順利通過，請執行 `git add .` 與 `git commit -m "refactor: use LangChain with_structured_output and Pydantic for robust LLM outputs"`。

請開始執行。