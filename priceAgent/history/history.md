# 修改歷史紀錄

## 2026-05-07 - Task 1-3 完成

### Task 1: 核心 Bug 修復

#### 1.1 修復 LLM JSON 解析錯誤
- **檔案**: `src/agents/price_guard.py`
- **修改**: `_parse_llm_response` 方法在 `json.loads` 前使用 Regex 清除 Markdown 標籤
- **代碼**:
```python
cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
data = json.loads(cleaned)
```

#### 1.2 修復無效的 Pydantic 驗證器
- **檔案**: `src/models/products.py`
- **修改**: `FraudDetectionResult` 使用 `@model_validator(mode="after")` 確保當 `suspicious_keywords` 非空時，`has_suspicious_text` 必為 True
- **代碼**:
```python
@model_validator(mode="after")
def validate_suspicious_text(self) -> "FraudDetectionResult":
    if len(self.suspicious_keywords) > 0 and not self.has_suspicious_text:
        return FraudDetectionResult(
            url=self.url,
            has_suspicious_text=True,
            suspicious_keywords=self.suspicious_keywords,
            price_analysis=self.price_analysis
        )
    return self
```

#### 1.3 修復資源生命週期 (Memory Leak)
- **檔案**: `src/agents/comparison_agent.py`, `src/main.py`
- **修改**: 
  - `ComparisonAgent` 實作為 Async Context Manager (`__aenter__`, `__aexit__`)
  - `main.py` 使用 `async with ComparisonAgent() as agent:` 呼叫

### Task 2: 非同步效能優化

#### 2.1 全面替換為非同步資料庫 (aiosqlite)
- **檔案**: `requirements.txt`, `src/utils/db.py`, `src/main.py`, `src/agents/comparison_agent.py`
- **修改**:
  - `requirements.txt` 加入 `aiosqlite>=0.19.0`
  - `db.py` 全面改為 async，使用 `aiosqlite.connect()` 和 `async with`
  - `main.py` 的 `show_history` 和 `show_stats` 改為異步呼叫
  - `comparison_agent.py` 的 `db.save_search_log` 和 `db.save_shopping_result` 加上 `await`

#### 2.2 加入爬蟲併發控制
- **檔案**: `src/agents/comparison_agent.py`
- **修改**: 在 `compare` 方法中使用 `asyncio.Semaphore(3)` 限制併發數量
- **代碼**:
```python
semaphore = asyncio.Semaphore(3)

async def _constrained_scrape(result):
    async with semaphore:
        return await self._scrape_and_analyze(result["url"], result["title"])

tasks = [_constrained_scrape(result) for result in search_results]
analyses = await asyncio.gather(*tasks, return_exceptions=True)
```

### Task 3: LLM 功能增強

#### 3.1 升級 PriceAnalysis 模型
- **檔案**: `src/models/products.py`
- **修改**: 新增兩個欄位
  - `purchase_reason: Optional[str]` - 購買理由與商品優勢
  - `price_confidence: int` - 真實價格信心度 (0-100)

#### 3.2 強化 LLM 價格萃取與驗證
- **檔案**: `src/agents/comparison_agent.py`
- **修改**: `_extract_price_with_llm` 方法
  - 回傳類型改為 `tuple[float, int]` (price, confidence)
  - Prompt 修改為要求 LLM 回傳 JSON 格式包含 `real_price` 與 `confidence`
  - 若信心度低於 50%，視為抓取失敗並返回初始價格
- **代碼**:
```python
prompt = f"""...
請以 JSON 格式回應，包含以下欄位：
- real_price: 商品的實際售價（數字），如果找不到則為 0
- confidence: 你對這個價格的信心度（0-100，基於價格是否在明顯的價格標示區域）
...
"""
```

#### 3.3 強化風險判斷與購買理由
- **檔案**: `src/agents/price_guard.py`
- **修改**: 
  - `analyze_with_llm` 的 Prompt 增加 `purchase_reason` 生成要求
  - `_parse_llm_response` 更新以解析新增的欄位
  - 確保 `price_analysis.price_confidence` 被正確設定

### 其他修改

#### CLAUDE.md 創建
- 創建專案指南文件，包含架構概述、關鍵模式、常用命令

#### requirements.txt 更新
- 移除版本號限制以解決相容性問題

#### git 初始化
- 初始化 git repository
- 完成初始提交
