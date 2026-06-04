# 你現在是一個資深的 Python 軟體工程師。
## 請幫我修改比價機器人專案中的 `src/agents/comparison_agent.py` 檔案。

【目前的運作流程】
-
目前的程式在 `compare` 函數中，會先呼叫 `_search_products` 取得 Brave 搜尋結果，然後直接對過濾後的前幾個結果進行 `_scrape_and_analyze` 網頁抓取。

【我要新增的優化功能：LLM 預先評分過濾】
-
為了避免浪費資源去抓取沒用的網頁，我希望在「拿到搜尋結果清單」之後，先請 LLM (ChatOllama) 根據搜尋結果的標題 (title)、網址 (url) 和描述 (description) 進行評分與排序，挑選出最可能是「真實商品販售頁面」的前 3 名，再派爬蟲去抓取這 3 個網址。

【具體實作要求】
-
1. 請在 `ComparisonAgent` 類別中，新增一個非同步函數，例如 `_rank_results_with_llm(self, query: str, search_results: list[dict]) -> list[dict]`。
2. 在這個新函數中，組合一個 Prompt，把 search_results 裡每個項目的標題、網址、描述列出來給 LLM 看。
3. 要求 LLM 以 JSON 格式回傳（例如只回傳前 3 名推薦的網址清單，或是給每個網址 1-100 的評分）。
4. 在 `compare` 函數中，呼叫這個新方法來取代原本直接拿 `search_results` 去抓取的邏輯。只針對 LLM 挑選出的前 3 名最高分網址去執行 `_scrape_and_analyze`。
5. 【超重要防呆機制】：這個 LLM 評分步驟必須加上超時限制 (例如 asyncio.wait_for 設定 15 秒) 以及完整的 try-except 錯誤處理。如果 LLM 回答格式壞掉、或是超時沒有回答，程式絕對不能崩潰，必須「自動退回原本的做法」，也就是直接拿清單的前 3 個網址去抓取。
6. 測試執行功能運作

- 請給我修改後的完整程式碼重點片段，並加上詳細的中文註解，確保程式邏輯穩定不出錯。