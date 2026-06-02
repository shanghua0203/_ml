# LSTM 中文語言模型（文字接龍）

使用 PyTorch LSTM 實作的中文文字生成模型，**絕對不使用 Transformer / Attention**。
就像一台「文字接龍機器」：給它一個開頭，它會自動接下去寫句子。

## 專案架構

```
.
├── text_processor.py   # 資料處理：jieba 分詞、字典、文字轉數字、DataLoader
├── model.py            # 模型架構：Embedding + LSTM(2層) + Linear（含 Weight Tying）
├── main.py             # 主程式：訓練迴圈 + 存讀檔 + 文字生成（Top-k + Temperature）
├── test_main.py        # 單元測試 + 系統測試
├── test.sh             # 一鍵測試腳本
├── requirements.txt    # Python 套件依賴清單
├── _doc/               # 版本歷史紀錄
│   ├── v0.1.md         # 初始版本
│   ├── v0.2.md         # 模組化
│   ├── v0.3.md         # 測試框架
│   ├── v0.4.md         # README 重寫
│   ├── v1.0.md         # 專業化升級
│   └── v1.1.md         # 詞層級 + Weight Tying + GPU + Top-k
├── dataset.txt         # 外部訓練資料（手沖咖啡教學文）
├── model_checkpoint.pt # 訓練好的模型權重檔
└── README.md           # 本檔案（繁體中文說明書）
```

## 資料流（資料是怎麼被加工的）

```
外部 .txt 或備用故事
  → load_external_text()       # 讀取文字檔
  → jieba.lcut()               # 把句子切成「詞」（不再是單一字）
  → build_vocab()              # 建立「詞→編號」字典（含 <UNK> 未知詞）
  → text_to_ids()              # 把詞轉成數字編號
  → prepare_training_data()    # 切成「輸入→正確答案」的訓練對
  → TextDataset + DataLoader   # 裝進批次推車，一次搬一批給模型
  → MyLanguageModel()          # LSTM 神經網路訓練
  → torch.save()               # 把訓練好的權心存成檔案
  → torch.load()               # 下次直接讀取，不用重訓
  → generate_text()            # 文字接龍（Top-k + Temperature）
  → 輸出結果
```

## 環境設定

### 1. 建立虛擬環境（就像幫專案準備一個專屬的房間）

```bash
python3 -m venv .venv
```

### 2. 安裝所需套件

```bash
.venv/bin/pip install -r requirements.txt
```

這樣就會一次裝好 PyTorch、jieba、pytest 三個套件。

如果沒有 `requirements.txt`，也可以手動裝：

```bash
.venv/bin/pip install torch jieba pytest
```

**注意**：jieba 是用來把中文句子切成「詞」的工具（就像把「我愛吃蘋果」切成「我 / 愛吃 / 蘋果」）。

## 執行方式

### 訓練模型 + 看生成結果

```bash
.venv/bin/python main.py
```

執行後會：
1. 自動讀取 `dataset.txt`（如果不存在就用內建備用故事）
2. 用 jieba 把文字切成詞
3. 建立詞典（含 `<UNK>` 未知詞標記）
4. 自動偵測你的電腦有沒有 GPU（CUDA）或 Apple Silicon（MPS），有的話就用，沒有就用 CPU
5. 訓練 LSTM 模型 150 個回合
6. 把訓練好的權心存到 `model_checkpoint.pt`
7. 展示文字生成結果（用 Top-k + Temperature 抽樣）

### 使用自己的訓練資料

把你的文字檔命名為 `dataset.txt` 放在專案根目錄：

```bash
echo "你的訓練文字放在這裡" > dataset.txt
.venv/bin/python main.py
```

如果找不到 `dataset.txt`，程式會自動用內建的 50 字備用故事，不會當機。

### 一鍵執行所有測試

```bash
chmod +x test.sh
./test.sh
```

### 手動執行測試

```bash
.venv/bin/python -m pytest test_main.py -v
```

## 模型超參數（可以調整的設定）

| 參數 | 預設值 | 這是什麼？像什麼？ |
|------|--------|-------------------|
| `embed_size` | 64 | 每個詞用幾個數字來表達意思（就像用 64 個重點來描述一個詞） |
| `hidden_size` | 64 | LSTM 的記憶容量（就像大腦可以記住 64 件事） |
| `num_layers` | 2 | LSTM 疊幾層（2 層就像兩個過濾網疊在一起） |
| `dropout` | 0.2 | 訓練時隨機忘掉 20% 的資訊，防止死背（就像老師遮住部分課本） |
| `sequence_length` | 8 | 用前面幾個詞來猜下一個詞（就像看前面 8 個字來猜第 9 個） |
| `batch_size` | 64 | 一次搬幾筆資料給模型（就像手推車一次載 64 箱貨） |
| `learning_rate` | 0.01 | 每次調整參數的幅度（太大了會跌跌撞撞，太小了學很慢） |
| `total_epochs` | 150 | 總共訓練幾回合 |
| `temperature` | 0.8 | 控制生成文字的隨機度（越低越保守，越高越有創意） |
| `top_k` | 5 | 只從最熱門的前 5 個詞中抽樣，避免選到奇怪的字 |

## v1.1 新功能介紹

### 1. 詞層級分詞（jieba）

**以前**：把「我愛吃蘋果」拆成「我 / 愛 / 吃 / 蘋 / 果」（5 個單字，看不懂意思）。
**現在**：把「我愛吃蘋果」拆成「我 / 愛吃 / 蘋果」（3 個詞，有意義）。

就像小學生一開始學認「字」，長大後學會認「詞」，理解得更完整。

### 2. <UNK> 未知詞防呆機制

字典裡會自動加入一個 `<UNK>` 記號（編號 0）。
如果遇到沒看過的詞，就用 `<UNK>` 代替，**不會當機**。
就像字典第一頁先畫一個「問號」，查不到的字都丟到那裡。

### 3. Weight Tying（權重共享）

讓 Embedding 層（把詞轉成數字）和 Linear 層（把數字轉回詞）共用同一組權重。
就像學生和老師共用同一本課本，不用一人一本，省空間又學得更好。

**注意**：啟用 Weight Tying 時，`embed_size` 會自動等於 `hidden_size`。

### 4. GPU / MPS 自動偵測

程式會自動檢查你的電腦有沒有：
- **CUDA**（NVIDIA 顯示卡）→ 用 GPU 加速
- **MPS**（Apple Silicon，如 M1/M2/M3 晶片）→ 用 Apple 晶片加速
- **都沒有** → 用 CPU，還是可以跑，只是比較慢

就像手機自動切換 Wi-Fi 和行動網路，有好的就用好的。

### 5. Top-k 採樣

生成文字時，只從機率最高的前 k 個詞中抽樣（預設 k=5）。
就像在超市選購時，先從貨架上挑出最熱門的前 5 種商品，再從中選擇，
不會浪費時間看整排貨架，也比較不會選到奇怪的商品。

## 模組說明

### `text_processor.py` — 資料處理小幫手

| 函數 / 類別 | 在做什麼？ |
|------------|-----------|
| `load_external_text(filepath)` | 讀取外部 .txt 檔案，找不到就用備用故事 |
| `tokenize(text)` | 用 jieba 把句子切成詞語串列 |
| `build_vocab(text)` | 建立「詞→編號」和「編號→詞」兩本字典（含 `<UNK>`） |
| `text_to_ids(text, word_to_id)` | 把中文句子轉成數字串列，沒看過的詞自動用 `<UNK>` |
| `prepare_training_data(ids, seq_len)` | 把數字串列切成「輸入→正確答案」的訓練對 |
| `TextDataset` | 把 inputs/targets 包成 PyTorch 認得的資料格式 |
| `create_dataloader(inputs, targets, batch_size, shuffle)` | 建立批次推車，一次搬一批資料 |
| `auto_device()` | 自動偵測 CUDA / MPS / CPU |

### `model.py` — 模型工廠

```
MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, num_layers=2, dropout=0.2, tie_weights=True)
```

- `forward(x, hidden_state)` → `(output, hidden_state)`
- 輸出形狀：`[batch_size, vocab_size]`
- 架構：Embedding → Dropout → LSTM(2層) → Linear
- 啟用 Weight Tying 時，Embedding 跟 Linear 共用權重

### `main.py` — 主程式

- 自動偵測硬體（CUDA → MPS → CPU）
- 使用 DataLoader 批次訓練
- 訓練後自動儲存權重至 `model_checkpoint.pt`
- 支援從存檔讀取權重，不用重新訓練
- `generate_text()` 支援 Top-k 採樣 + Temperature

## 進階功能說明

### 存檔 / 讀檔機制

訓練完的模型可以存成檔案，下次直接讀取，不用重訓：

```python
torch.save(model.state_dict(), "model_checkpoint.pt")
model.load_state_dict(torch.load("model_checkpoint.pt", weights_only=True))
```

就像考試前把重點存成 PDF，下次考試直接拿出來看。

### Top-k 採樣 + Temperature

兩層過濾機制，讓文字生成更穩定又有創意：

1. **Top-k 過濾**：先砍掉機率低的選項，只留前 k 個
2. **Temperature 縮放**：調整剩下選項的「尖銳程度」
3. **Softmax + 抽籤**：根據機率抽出一個詞

| Temperature | 效果 | 比喻 |
|-------------|------|------|
| 0.3 | 非常保守，幾乎固定 | 冬天，大家都擠在一起 |
| 0.8 | 稍微保守，仍有變化（預設） | 春天，有點涼但舒適 |
| 1.0 | 照原始機率抽樣 | 夏天，一切正常 |
| 1.5 | 很有創意，隨機度高 | 秋天，大家一起散開 |

### Weight Tying（權重共享）

Embedding 層（把詞編號轉成向量）和 Linear 輸出層（把向量轉回詞）共用同一組權重。
優點：
- **減少模型體積**：少了一組大型權重矩陣
- **學得更好**：輸入端和輸出端用同一套「詞的表示法」
- **像兩個人共用同一本字典**，溝通起來更有效率

### <UNK> 未知詞處理

如果遇到字典裡沒有的詞，程式不會當機，而是用 `<UNK>` 代替。
就像老師發考卷時，看不懂的單字先用紅筆圈起來標記。
