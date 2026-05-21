# LSTM 中文語言模型

使用 PyTorch LSTM 實作的中文文字生成模型，**不使用 Transformer / Attention**。

## 專案架構

```
.
├── text_processor.py   # 資料處理：字典、文字轉數字、Dataset/DataLoader、外部資料讀取
├── model.py            # 模型架構：Embedding + Dropout + LSTM(2層) + Linear
├── main.py             # 主程式：訓練迴圈 + 存檔/讀檔 + 文字生成（含 Temperature）
├── test_main.py        # 單元測試 + 系統測試（72 項）
├── test.sh             # 一鍵測試腳本
├── README.md           # 本檔案
└── .venv/              # Python 虛擬環境（含 PyTorch）
```

## 資料流

```
外部 .txt 或備用故事 → load_external_text()
                     → build_vocab() → 字典 (char↔id)
                     → text_to_ids() → 數字序列
                     → prepare_training_data() → (inputs, targets)
                     → TextDataset + DataLoader → 批次資料
                     → MyLanguageModel() → 訓練 → torch.save() 存檔
                     → torch.load() 讀檔 → generate_text(temperature) → 生成結果
```

## 環境設定

### 建立虛擬環境

```bash
python3 -m venv .venv
```

### 安裝依賴

```bash
.venv/bin/pip install torch pytest
```

## 執行

### 訓練與生成

```bash
.venv/bin/python main.py
```

### 使用外部資料

將文字檔案命名為 `dataset.txt` 放在專案根目錄，程式會自動讀取：

```bash
echo "你的訓練文字放在這裡" > dataset.txt
.venv/bin/python main.py
```

如果找不到 `dataset.txt`，會自動使用內建的 50 字備用故事。

### 一鍵測試

```bash
chmod +x test.sh
./test.sh
```

### 手動執行測試

```bash
.venv/bin/python -m pytest test_main.py -v
```

## 模型超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `embed_size` | 32 | 字向量維度 |
| `hidden_size` | 64 | LSTM 隱藏層維度 |
| `num_layers` | 2 | LSTM 層數（Dropout 需要至少 2 層） |
| `dropout` | 0.2 | 隨機遺忘比例，防止過度擬合 |
| `sequence_length` | 8 | 上下文長度 |
| `batch_size` | 4 | DataLoader 批次大小 |
| `learning_rate` | 0.01 | Adam 優化器學習率 |
| `total_epochs` | 150 | 訓練輪數 |
| `temperature` | 0.8 | 文字生成溫度（越低越保守，越高越有創意） |

## 模組說明

### `text_processor.py`

| 函數/類別 | 說明 |
|-----------|------|
| `load_external_text(filepath)` | 讀取外部 .txt，找不到則使用備用故事 |
| `build_vocab(text)` | 建立 `(char_to_id, id_to_char, vocab_size)` |
| `text_to_ids(text, char_to_id)` | 字串 → 數字串列 |
| `prepare_training_data(ids, seq_len)` | 數字串列 → `(inputs, targets)` |
| `TextDataset` | PyTorch Dataset，包裝 inputs/targets 為 Tensor |
| `create_dataloader(inputs, targets, batch_size, shuffle)` | 建立 DataLoader 批次推車 |

### `model.py`

`MyLanguageModel(vocab_size, embed_size=32, hidden_size=64, num_layers=2, dropout=0.2)`

- `forward(x, hidden_state)` → `(output, hidden_state)`
- 輸出形狀：`[batch_size, vocab_size]`
- 架構：Embedding → Dropout → LSTM(2層) → Linear

### `main.py`

- 嘗試讀取 `dataset.txt`，找不到則使用備用故事
- 使用 DataLoader 批次訓練（batch_size=4）
- 訓練後自動儲存權重至 `model_checkpoint.pt`
- 支援從存檔讀取權重，避免重新訓練
- `generate_text()` 支援 Temperature 參數控制生成創意度

## 進階功能

### 存檔/讀檔機制

訓練完成後自動儲存 `model_checkpoint.pt`，下次執行可讀取權重：

```python
torch.save(model.state_dict(), "model_checkpoint.pt")
model.load_state_dict(torch.load("model_checkpoint.pt", weights_only=True))
```

### Temperature 採樣

控制文字生成的隨機度與創意度：

| Temperature | 效果 |
|-------------|------|
| 0.3 | 非常保守，幾乎只選機率最高的字 |
| 0.8 | 稍微保守，仍有合理變化（預設） |
| 1.0 | 按照原始機率抽樣 |
| 1.5 | 更有創意，機率低的字也有機會 |

### Dropout 防止過度擬合

LSTM 設為 2 層，層間 dropout=0.2，訓練時隨機遺忘 20% 的神經元，讓模型學會真正理解上下文而非死背訓練資料。
