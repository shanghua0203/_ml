# 本地端 AI 比價購物助理

## 專案目標

本專案是一個完全在本地端運行的 AI 比價購物助理，利用 Ollama 的 Llama 3 模型作為大腦，結合 Brave Search API 與 Crawl4AI 網頁抓取技術，提供智能比價服務。系統具備防偽過濾功能，能識別價格異常與詐騙特徵，為使用者推薦最適合的購買選擇。

## 核心功能

- **智能搜尋**：呼叫 Brave API 取得購物網站連結（24 小時快取機制）
- **網頁解析**：使用 Crawl4AI 抓取動態網頁並轉換為 Markdown
- **防偽過濾**：
  - 價格守門員：標記過低價格為風險
  - 文案雷達：掃描高仿、拆機、詐騙特徵關鍵字
- **智能推薦**：比較後輸出「最推薦」、「最便宜」與「最安全」三個選項
- **本地儲存**：SQLite 資料庫儲存歷史比價紀錄

## 專案結構

```
claude_ML_Project/
├── .env.example              # 環境變數範例
├── requirements.txt          # Python 依賴套件
├── src/
│   ├── __init__.py
│   ├── main.py              # 主程式與 CLI
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # 設定與環境變數管理
│   │   └── cache.py         # 快取機制
│   ├── models/
│   │   ├── __init__.py
│   │   └── products.py      # Pydantic 資料模型
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── scrape_agent.py  # 網頁抓取 Agent
│   │   ├── price_guard.py   # 價格守門員 Agent
│   │   └── comparison_agent.py  # 比價比較 Agent
│   └── utils/
│       ├── __init__.py
│       ├── db.py            # 資料庫操作
│       └── scoring.py       # 評分引擎
├── data/                    # 資料庫檔案（自動建立）
└── tests/                   # 測試檔案
```

## 環境設定與安裝

### 系統需求

- Python 3.10 或更高版本
- 本地端運行 Ollama 服務（預設 port 11434）
- Brave Search API 金鑰

### 安裝步驟

#### 1. 建立虛擬環境

```bash
# 建立虛擬環境
python3 -m venv .venv

# 啟用虛擬環境
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\activate
```

#### 2. 安裝依賴套件

```bash
# 升級 pip
pip install --upgrade pip

# 安裝套件
pip install -r requirements.txt
```

#### 3. 設定環境變數

```bash
# 複製範例檔案
cp .env.example .env

# 編輯 .env 檔案，填入 Brave API 金鑰
vi .env  # 或使用其他編輯器
```

`.env` 檔案內容：
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
BRAVE_API_KEY=your_brave_api_key_here
```

#### 4. 安裝 Ollama 與 Llama 3 模型

```bash
# 下載並安裝 Ollama
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# 啟動 Ollama 服務
ollama serve

# 在另一個終端機安裝 Llama 3 模型
ollama pull llama3
```

### Windows PowerShell 注意事項

```powershell
# 1. 啟用執行政策（如需要）
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. 建立虛擬環境
python -m venv .venv

# 3. 啟用虛擬環境
.\.venv\Scripts\Activate.ps1

# 4. 安裝套件
pip install -r requirements.txt

# 5. 執行程式
python src\main.py --search "your query"
```

## 執行方式

### 基本比價

```bash
# 啟用虛擬環境
source .venv/bin/activate  # 或 Windows 的 .venv\Scripts\activate

# 搜尋商品
python src/main.py --search "iphone 15"
python src/main.py -s "MacBook Pro M3"
```

### JSON 輸出

```bash
python src/main.py --search "airpods pro" --json
```

### 檢視歷史紀錄

```bash
python src/main.py --history
```

### 檢視統計資料

```bash
python src/main.py --stats
```

### 單一網頁抓取測試

```bash
python src/main.py --scrape https://example.com
```

### 清除過期快取

```bash
python src/main.py --clear-cache
```

### 查看環境設定

```bash
python src/main.py --env
```

## 輸出結果

比價結果包含三個推薦：

```
============================================================
比價結果：iphone 15
============================================================

🏆 最推薦
   名稱：iPhone 15 官方旗艦店
   價格：NT$32,900
   平台：momo
   風險：low
   安全分數：95/100

💰 最便宜
   名稱：iPhone 15 限定特價
   價格：NT$29,900
   平台：shopee

🛡️  最安全
   名稱：iPhone 15 專賣店
   價格：NT$31,500
   風險：low
   安全分數：98/100
```

## 技術棧與工具

| 工具/框架 | 用途 |
|-----------|------|
| **Python 3.10+** | 執行環境 |
| **Ollama + Llama 3** | 本地端 LLM 大腦 |
| **LangChain** | Agent 流程串接 |
| **Brave Search API** | 商品關鍵字搜尋 |
| **Crawl4AI** | 動態網頁抓取與 Markdown 轉換 |
| **Pydantic V2** | 資料驗證與模型定義 |
| **SQLite** | 本地端資料庫 |

## 犯罪防護機制

- 價格過低檢測（低於門檻 50% 標記為高風險）
- 詐騙關鍵字掃描（高仿、拆機、僅供參考等）
- 輸出結果包含安全等級與分數

## 注意事項

1. Ollama 服務必須在本地運行，預設 port 11434
2. 需取得 Brave Search API 金鑰並設定環境變數
3. 首次執行會自動建立 `data/shopping.db` 資料庫
4. 搜尋結果快取 24 小時，避免重複 API 呼叫

## 後續改進方向

- 實際整合 Brave Search API（目前為模擬結果）
- 完善網頁價格解析器
- 增加更多購物平台支援
- 加強 LLM 分析能力
