# 問答對資料集預處理工具

## 概述

本工具將 `output.jsonl`（格式：`{"prompt": "...", "completion": "..."}`）轉換為經過分詞（Tokenization）的資料集，可直接用於 Hugging Face `Trainer` 或 LLaMA-Factory 等微調框架。

## 流程

```
output.jsonl  ──→  套用對話模板 (Chat Template)  ──→  分詞 (Tokenization)  ──→  tokenized_dataset/
```

## 環境需求

- Python 3.10+
- `transformers`
- `datasets`
- `jinja2`

```bash
pip install transformers datasets jinja2
```

## 使用方法

### 基本執行

```bash
python preprocess_dataset.py
```

### 設定參數

編輯 `preprocess_dataset.py` 頂部的變數：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `INPUT_FILE` | `output.jsonl` | 輸入的 JSONL 檔案路徑 |
| `OUTPUT_DIR` | `tokenized_dataset` | 輸出資料夾名稱 |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | 分詞器模型名稱或本地路徑 |
| `MAX_LENGTH` | `2048` | 單筆序列最大 Token 長度（超出截斷） |

### 自訂分詞器

若使用其他模型（如 Llama、Gemma），修改 `MODEL_NAME` 即可：

```python
MODEL_NAME = "google/gemma-2-2b-it"       # Gemma
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Llama
MODEL_NAME = "/path/to/local/tokenizer"   # 本地分詞器路徑
```

## 輸出格式

`tokenized_dataset/` 資料夾包含：

```
tokenized_dataset/
├── data-00000-of-00001.arrow    # 分詞後的資料（input_ids, attention_mask, length）
├── dataset_info.json            # 資料集中繼資料
└── state.json                   # 狀態資訊
```

使用 Hugging Face `datasets` 載入：

```python
from datasets import load_from_disk
dataset = load_from_disk("tokenized_dataset")
# dataset[0] → {"input_ids": [...], "attention_mask": [...], "length": 87}
```

## 與微調框架對接

### LLaMA-Factory

將 `tokenized_dataset` 目錄路徑填入 `dataset_dir`，或將資料複製到 LLaMA-Factory 的 `data/` 目錄下。

### Hugging Face Trainer

```python
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, Trainer

dataset = load_from_disk("tokenized_dataset")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # ...
)
trainer.train()
```

## 統計數據

執行完畢後會自動印出：

- 總問答筆數
- 每筆平均 Token 數
- 每筆最多 Token 數
- 總 Token 數
