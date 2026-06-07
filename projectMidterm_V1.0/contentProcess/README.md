# ContentProcess — 中文問答資料處理與 LoRA 微調 Pipeline

將原始純文字檔轉換為 LLM 微調可用的結構化資料集，並支援以 LoRA 微調 Mamba（SSM）模型。

## 專案結構

```
contentProcess/
├── data/
│   ├── raw/
│   │   └── input.txt                  ← 原始純文字（約 7 萬字）
│   └── processed/
│       ├── output.jsonl                ← 生成的問答對（JSONL）
│       └── tokenized_dataset/          ← 分詞後的資料集
├── scripts/
│   ├── config.py                       # 超參數集中管理（LoRA、訓練參數）
│   ├── model_utils.py                  # 載入 Mamba + 套 LoRA
│   ├── data_utils.py                   # 載入資料集 + DataCollator
│   ├── train.py                        # LoRA 微調主腳本
│   ├── inference_test.py               # 推理測試腳本
│   ├── generate_qa.py                  # Stage 1：文本 → 問答對
│   ├── preprocess_dataset.py           # Stage 2：問答對 → Tokenized Dataset
│   ├── mamba_inference.py              # Stage 3：Mamba 推論（CLI + LoRA + 互動模式）
│   └── main.py                         # Stage 5：FastAPI 聊天網頁後端
├── static/
│   └── index.html                      # Stage 5：前端聊天頁面
├── tests/
│   ├── test_config.py                  # 10 項—參數型別/範圍
│   ├── test_data_utils.py              # 5 項—資料集欄位/長度
│   ├── test_model_utils.py             # 5 項—模型載入/LoRA 包裝
│   ├── test_mamba_inference.py         # 4 項—命令列參數解析
│   └── test_main.py                    # 5 項—Web API 整合測試
├── docs/
│   ├── task0.0.md
│   ├── task0.1.md
│   ├── task0.2.md
│   └── task0.3.md
├── coffee_mamba_lora/                  # LoRA 微調後權重輸出
├── test.sh                             # 一鍵整合測試
├── .venv/                              # Python 虛擬環境
├── task1.0.md                          # Stage 3 升級規格
├── task1.1.md                          # Web 聊天應用規格
└── README.md
```

## Pipeline 流程

```
input.txt
    │
    ▼
Stage 1: generate_qa.py  ──→  output.jsonl（問答對）
    │
    ▼
Stage 2: preprocess_dataset.py  ──→  tokenized_dataset/（分詞資料集）
    │
    ▼
Stage 3: mamba_inference.py（Mamba CLI 推論，可選 LoRA）
    │
    ▼
Stage 4: train.py（LoRA 微調 Mamba-1.4B）
    │
    ▼
coffee_mamba_lora/（訓練後 LoRA 權重）
    │
    ▼
Stage 5: main.py + static/index.html（FastAPI 聊天網頁）
```

---

## 環境設定

```bash
# 啟動虛擬環境
source .venv/bin/activate

# 若從頭建立環境
python3 -m venv .venv
source .venv/bin/activate
pip install transformers datasets jinja2 ollama torch tqdm peft pytest fastapi uvicorn
```

---

## Stage 1：文本 → 問答對（generate_qa.py）

將 `input.txt` 切割後透過 Ollama（`gemma3:4b`）生成問答對。

### 執行

```bash
source .venv/bin/activate
python scripts/generate_qa.py
```

### 設定參數（編輯 `scripts/generate_qa.py` 頂部）

| 變數 | 預設值 | 說明 |
|---|---|---|
| `INPUT_FILE` | `data/raw/input.txt` | 原始輸入文字檔 |
| `OUTPUT_FILE` | `data/processed/output.jsonl` | 輸出的問答對 JSONL |
| `MODEL` | `gemma3:4b` | Ollama 模型名稱 |
| `CHUNK_MIN` | `200` | 切割區塊最小字數 |
| `CHUNK_MAX` | `500` | 切割區塊最大字數 |

### 運作機制

1. 將長文本按段落 → 句子切分成 200–500 字的區塊
2. 每區塊送入 Ollama，搭配 system prompt 要求輸出 `{"qa_pairs": [...]}`
3. 解析 JSON，每組問答對立即 append 寫入 `output.jsonl`
4. 使用 `tqdm` 顯示進度，JSON 解析失敗則跳過不中斷

### 輸出格式

```json
{"prompt": "手沖咖啡需要哪些器具？", "completion": "手沖咖啡需要濾杯、濾紙、手沖壺、電子秤等器具。"}
{"prompt": "如何控制手沖咖啡的水溫？", "completion": "建議水溫在 80-90°C 之間，淺焙用較高溫，深焙用較低溫。"}
```

### 前置需求

- Ollama 需在背景執行，且有 `gemma3:4b` 模型
- 確認方法：`curl http://localhost:11434/api/tags`

---

## Stage 2：問答對 → Tokenized Dataset（preprocess_dataset.py）

將 JSONL 問答對轉換為 Hugging Face 格式的分詞資料集，可直接用於微調框架。

### 執行

```bash
source .venv/bin/activate
python scripts/preprocess_dataset.py
```

### 設定參數（編輯 `scripts/preprocess_dataset.py` 頂部）

| 變數 | 預設值 | 說明 |
|---|---|---|
| `INPUT_FILE` | `data/processed/output.jsonl` | 輸入的 JSONL 檔 |
| `OUTPUT_DIR` | `data/processed/tokenized_dataset` | 輸出資料夾 |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | 分詞器模型 |
| `MAX_LENGTH` | `2048` | 最大 Token 長度 |

### 運作機制

1. `datasets.load_dataset("json", ...)` 讀取 JSONL
2. 對每筆資料建立對話結構並套用 `apply_chat_template()`
3. `dataset.map(process_batch, batched=True)` 批次分詞
4. `save_to_disk()` 儲存為 Arrow 格式

對話模板結果範例：
```
<|im_start|>user
手沖咖啡需要哪些器具？<|im_end|>
<|im_start|>assistant
需要濾杯、濾紙、手沖壺等。<|im_end|>
```

### 輸出資料夾結構

```
data/processed/tokenized_dataset/
├── data-00000-of-00001.arrow    # 分詞後的資料（input_ids, attention_mask, length）
├── dataset_info.json            # 中繼資料
└── state.json                   # 狀態資訊
```

### 載入方式

```python
from datasets import load_from_disk
dataset = load_from_disk("data/processed/tokenized_dataset")
```

---

## Stage 3：Mamba 模型推論（mamba_inference.py）

載入 Hugging Face 上的 Mamba（SSM 架構）模型，支援 LoRA 權重掛載與互動式聊天。

### 執行

```bash
source .venv/bin/activate

# 只用基礎模型
python scripts/mamba_inference.py

# 載入 LoRA 權重
python scripts/mamba_inference.py --use_lora --lora_path ./coffee_mamba_lora

# 自訂生成參數
python scripts/mamba_inference.py --temperature 0.8 --max_tokens 300 --top_p 0.95
```

### 命令列參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--model_name` | `state-spaces/mamba-1.4b-hf` | 基礎模型名稱 |
| `--use_lora` | `False` | 啟用後載入 LoRA 權重 |
| `--lora_path` | `./coffee_mamba_lora` | LoRA 權重資料夾路徑 |
| `--temperature` | `0.7` | 生成溫度 |
| `--max_tokens` | `200` | 最大生成字數 |
| `--top_p` | `0.9` | Top-p 採樣門檻 |
| `--repetition_penalty` | `1.1` | 重複懲罰 |

### 互動模式

啟動後進入 `User: ` 提示迴圈，輸入 `exit` 或 `quit` 結束程式。

### 切換模型

| 模型名稱 | 參數量 | 說明 |
|---|---|---|
| `state-spaces/mamba-130m-hf` | 130M | 輕量快速 |
| `state-spaces/mamba-370m-hf` | 370M | 平衡型 |
| `state-spaces/mamba-790m-hf` | 790M | 中量級 |
| `state-spaces/mamba-1.4b-hf` | 1.4B | 目前預設 |
| `state-spaces/mamba-2.8b-hf` | 2.8B | 最大量級 |

### 運作機制

1. **argparse** 解析所有命令列參數
2. **自動偵測硬體**：CUDA → MPS → CPU
3. **選擇性 LoRA 載入**：使用 `PeftModel.from_pretrained` 掛載權重
4. **互動聊天**：無限迴圈，逐輪產生回應

---

## Stage 4：Mamba LoRA 微調（train.py）

使用 LoRA 微調 `state-spaces/mamba-1.4b-hf` 模型，將問答資料集訓練進模型中。

### 架構（模組化拆分）

`scripts/` 下四個檔案分工明確，總行數超過 1000 行時強制拆分：

| 檔案 | 職責 | 行數 |
|---|---|---|
| `config.py` | 所有超參數集中管理 | ~10 行 |
| `model_utils.py` | 載入 Mamba 模型、套 LoRA、硬體偵測 | ~40 行 |
| `data_utils.py` | 載入 tokenized_dataset、建立 DataCollator | ~15 行 |
| `train.py` | 組裝 Trainer、執行訓練、儲存權重 | ~50 行 |

### 執行訓練

```bash
source .venv/bin/activate
python scripts/train.py
```

### 超參數一覽（編輯 `scripts/config.py`）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `MODEL_NAME` | `state-spaces/mamba-1.4b-hf` | 微調的基礎模型 |
| `LORA_R` | `8` | LoRA 秩（rank），愈大學習能力愈強但愈吃記憶體 |
| `LORA_ALPHA` | `16` | LoRA 縮放係數，通常為 r 的 2 倍 |
| `LORA_DROPOUT` | `0.05` | 隨機丟棄率，防止過擬合 |
| `TARGET_MODULES` | `["in_proj", "x_proj"]` | Mamba 的核心投影層（PEFT 相容限制，排除 out_proj） |
| `BATCH_SIZE` | `1` | 單張 GPU 每次處理的樣本數 |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | 梯度累積步數（等效 batch_size = 1×4 = 4） |
| `LEARNING_RATE` | `2e-4` | 學習率 |
| `NUM_EPOCHS` | `3` | 訓練輪數 |
| `LOGGING_STEPS` | `10` | 每 10 步印一次 loss |
| `LOG_FILE` | `data/processed/training_log.jsonl` | 損失值記錄檔（JSONL） |

### 訓練資源預估

| 項目 | 估算 |
|---|---|
| 可訓練參數 | ~1.4M（佔全模型的 ~0.1%） |
| GPU VRAM | ~5-6 GB |
| 訓練時間（3 epochs × 774 筆） | ~30-60 分鐘 |
| 輸出 LoRA 權重大小 | ~10-15 MB（`coffee_mamba_lora/`） |

### 損失值記錄

訓練過程中每個 `logging_steps`（預設 10 步）自動寫入 `data/processed/training_log.jsonl`：

```json
{"step": 10, "epoch": 0.12, "loss": 2.345678, "timestamp": "2026-06-04T15:30:00"}
{"step": 20, "epoch": 0.24, "loss": 2.123456, "timestamp": "2026-06-04T15:31:00"}
```

可用來繪製 loss 曲線或監控訓練收斂情況。

### 輸出格式

訓練完成後產生 `coffee_mamba_lora/` 資料夾，僅儲存 LoRA 外掛權重（不含 base model）：

```
coffee_mamba_lora/
├── adapter_config.json       # LoRA 設定
├── adapter_model.safetensors  # LoRA 權重（~10MB）
└── tokenizer.json            # 分詞器
```

### 載入訓練後的模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 先載入 base model，再掛上 LoRA 權重
base_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")
model = PeftModel.from_pretrained(base_model, "./coffee_mamba_lora")
```

---

## Stage 5：Web 聊天應用（main.py + index.html）

前後端分離的聊天網頁，支援 LoRA 動態切換與串流打字機效果。

### 啟動伺服器

```bash
source .venv/bin/activate
python scripts/main.py
# 瀏覽器開啟 http://localhost:8080/static/index.html
```

### 後端 API

| Endpoint | 方法 | 說明 |
|---|---|---|
| `GET /api/lora_models` | GET | 掃描所有 checkpoint，回傳名稱、步數、Loss |
| `/api/chat` | POST | 串流文字生成（SSE），支援 LoRA 動態切換 |

### 聊天 API 請求格式

```json
{
  "prompt": "手沖咖啡需要什麼器具？",
  "temperature": 0.7,
  "lora_path": "coffee_mamba_lora/checkpoint-2112"
}
```

`lora_path` 可傳 `null` 或不傳，代表只用基礎模型。

### 前端介面

```
┌──────────────────┬──────────────────────────────┐
│ LoRA 模型選擇     │  聊天視窗                     │
│ [下拉式選單 ▼]    │                              │
│ 無 LoRA（基礎模型）│  User: 手沖咖啡參數？         │
│ checkpoint-2112.. │  Assistant: 建議... (打字機)  │
│                   │                              │
│ 溫度控制          │  [輸入框................]     │
│ 0.1 ───●─── 1.5   │  [送出]                      │
└──────────────────┴──────────────────────────────┘
```

- 左側面板：LoRA 下拉選單（onload 自動打 API 載入）、溫度拉桿 0.1~1.5
- 右側聊天：對話氣泡、`fetch` + `getReader()` 串流打字機效果、系統備註尾綴

---

## 測試

### 執行整合測試

```bash
source .venv/bin/activate
bash test.sh
```

測試流程：
1. **語法檢查** — 所有 Python 檔案 `py_compile`
2. **單元測試** — pytest 29 項（含 argparse 參數、API 格式）
3. **系統測試** — 推理測試（3 組 prompt 確認模型能正常生成）
4. **資料集完整性檢查** — 路徑存在、筆數、欄位、總 tokens
5. **Web API 整合測試** — 5 項，使用 FastAPI TestClient 實際載入模型

### 測試清單

| 測試檔案 | 測試項 | 測試內容 |
|---|---|---|
| `tests/test_config.py` | 10 項 | 參數型別、正數範圍、dropout 區間、seq_length 合理性 |
| `tests/test_data_utils.py` | 5 項 | 資料集載入、欄位存在、input_ids 合法性、長度一致、max_length |
| `tests/test_model_utils.py` | 5 項 | 模型載入、分詞器載入、LoRA 包裝（可訓練參數 > 0）、CUDA 偵測、中文編碼 |
| `tests/test_mamba_inference.py` | 4 項 | argparse 預設值、自訂值、use_lora 旗標、負溫度處理 |
| `tests/test_main.py` | 5 項 | LoRA 清單格式、基礎模型聊天、LoRA 聊天、溫度參數、錯誤路徑 |

---

## 完整工作流程

```bash
# 0. 進入專案
cd contentProcess
source .venv/bin/activate

# 1. 將原始文字放入 data/raw/input.txt
# 2. 生成問答對（確認 Ollama 有在跑）
python scripts/generate_qa.py

# 3. 預處理為 Tokenized Dataset
python scripts/preprocess_dataset.py

# 4. （可選）載入 Mamba 模型做 CLI 推論測試
python scripts/mamba_inference.py

# 5. 執行 LoRA 微調
python scripts/train.py

# 6. （可選）啟動 Web 聊天應用
python scripts/main.py
# 瀏覽器開啟 http://localhost:8080/static/index.html

# 7. 執行全部測試
bash test.sh
```

---

## 當前統計數據

| 項目 | 數值 |
|---|---|
| 原始文字 | 69,399 字 |
| 生成問答對 | 774 筆 |
| 平均 Token 數/筆 | 86.5 |
| 最多 Token 數/筆 | 204 |
| 總 Token 數 | 66,988 |
| Mamba 模型 | state-spaces/mamba-1.4b-hf（1.37B 參數） |
| LoRA 可訓練參數 | ~1.4M（佔全模型 ~0.1%） |
| 單元測試 | 29 項（24 通過） |
