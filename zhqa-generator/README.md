# zhqa-generator

> 將中文純文字切割並透過 Ollama LLM 自動生成 QA 問答對，供 LLM 微調使用。

[![PyPI version](https://img.shields.io/pypi/v/zhqa-generator)](https://pypi.org/project/zhqa-generator/)
[![Python](https://img.shields.io/pypi/pyversions/zhqa-generator)](https://pypi.org/project/zhqa-generator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 安裝

```bash
pip install zhqa-generator
```

> **前置需求**：需要在本機執行 [Ollama](https://ollama.com/) 並拉取所需模型：
> ```bash
> ollama pull gemma3:4b
> ```

---

## 快速開始

### Python API

```python
from zhqa_generator import chunk_text, generate_qa_pairs

# 讀取文章
text = open("article.txt", encoding="utf-8").read()

# Step 1：切割文本（每塊 200~500 字）
chunks = chunk_text(text, chunk_min=200, chunk_max=500)
print(f"共切割成 {len(chunks)} 個區塊")

# Step 2：呼叫 Ollama 生成 QA 對
pairs = generate_qa_pairs(chunks, model="gemma3:4b")

# Step 3：儲存為 JSONL
import json
with open("output.jsonl", "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"共生成 {len(pairs)} 組問答對")
```

輸出範例：
```json
{"prompt": "手沖咖啡需要哪些器具？", "completion": "手沖咖啡需要濾杯、濾紙、手沖壺、電子秤等器具。"}
{"prompt": "如何控制手沖咖啡的水溫？", "completion": "建議水溫在 80-90°C 之間，淺焙用較高溫，深焙用較低溫。"}
```

### 命令列（CLI）

```bash
# 基本用法
zhqa-generate --input article.txt --output output.jsonl

# 自訂模型與切割大小
zhqa-generate --input article.txt --output output.jsonl \
              --model llama3:8b \
              --chunk-min 300 --chunk-max 600
```

#### CLI 參數一覽

| 參數 | 短名 | 預設值 | 說明 |
|---|---|---|---|
| `--input` | `-i` | （必填） | 輸入純文字檔 |
| `--output` | `-o` | `output.jsonl` | 輸出 JSONL 檔路徑 |
| `--model` | `-m` | `gemma3:4b` | Ollama 模型名稱 |
| `--chunk-min` | — | `200` | 區塊最小字元數 |
| `--chunk-max` | — | `500` | 區塊最大字元數 |

---

## API 參考

### `chunk_text(text, chunk_min=200, chunk_max=500) → list[str]`

將長文本依段落、句子智慧切割為適合 LLM 的區塊。

### `generate_qa_for_chunk(chunk, model="gemma3:4b", system_prompt=...) → list[dict]`

對單一區塊呼叫 Ollama，回傳 `[{"prompt": ..., "completion": ...}]`。

### `generate_qa_pairs(chunks, model="gemma3:4b", show_progress=True) → list[dict]`

批次處理所有區塊，附帶 tqdm 進度條。

---

## 使用情境

- 將書籍、文章、論文轉換為 LLM 微調資料集（JSONL 格式）
- 搭配 `transformers` + `peft` 做 LoRA 微調
- 快速產生繁體中文問答對做評測資料

---

## License

[MIT](LICENSE)
