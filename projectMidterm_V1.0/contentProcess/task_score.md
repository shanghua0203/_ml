# 角色設定
你現在是一位專業的 Python 資料科學家與深度學習工程師。我需要你幫我撰寫一個 Python 腳本（例如命名為 `analyze_training_logs.py`），用來分析我微調 Mamba 模型時產生的 JSONL 訓練日誌檔，並繪製出專業、符合學術期末報告水準的圖表。

# 背景與資料格式
我的訓練日誌檔路徑為 `data/processed/training_log.jsonl`，每一行是一個 JSON 物件，記錄了每 10 步 (logging steps) 的訓練狀態。
範例資料如下：
`{"step": 10, "epoch": 0.12, "loss": 2.345678, "timestamp": "2026-06-04T15:30:00.000000"}`
`{"step": 20, "epoch": 0.24, "loss": 2.123456, "timestamp": "2026-06-04T15:31:00.000000"}`

# 任務需求
請使用 `pandas`, `matplotlib`, `seaborn`, `datetime` 等標準套件，讀取上述 JSONL 檔案，並計算與繪製以下四個維度的數據。請將這四張圖繪製在同一個畫布上（例如 2x2 的 Subplots），或是分別存成四張高解析度圖片：

1. **訓練損失曲線 (Training Loss)**
   - X 軸：Step 或 Epoch
   - Y 軸：Loss
   - 邏輯：直接讀取 JSON 中的 `loss` 欄位繪製平滑曲線。

2. **訓練困惑度曲線 (Training Perplexity, PPL)**
   - X 軸：Step 或 Epoch
   - Y 軸：Perplexity
   - 邏輯：由 Loss 計算得出，公式為 `PPL = math.exp(loss)`。請注意 Y 軸範圍，若初期 PPL 過高可考慮對 Y 軸取 log scale，或截斷極端值以保持圖表可讀性。

3. **學習率變化曲線 (Learning Rate Schedule)**
   - X 軸：Step
   - Y 軸：Learning Rate
   - 邏輯：日誌中沒有記錄 LR，請依據 HuggingFace Trainer 的預設線性衰減 (Linear Decay) 進行數學模擬。
   - 參數設定：已知初始學習率 (Initial LR) 為 `2e-4`，最終學習率為 `0`。
   - 總步數估算：可以透過讀取日誌中最大的 step 與其對應的 epoch，推算出 `Total_Steps = max_step / max_epoch * 3` (假設總訓練輪數為 3 epochs)，並以此繪製線性下降的直線。

4. **訓練速度與時間分析 (Training Speed)**
   - X 軸：Step
   - Y 軸：處理時間 (秒 / 每 10 步)
   - 邏輯：解析 ISO 格式的 `timestamp`，計算相鄰兩筆紀錄（即每 10 steps）的時間差（以秒為單位）。請繪製這個時間差的折線圖或散佈圖，並計算出「平均每步耗時 (Seconds per step)」。

# 程式碼輸出規範
1. 程式碼需具備模組化與良好的註解。
2. 圖表風格請設定為學術專業風格（如 `sns.set_theme(style="whitegrid")`）。
3. 產生圖表後，除了顯示 (plt.show) 之外，請將圖片儲存至 `data/processed/training_analysis.png`。
4. 程式執行完畢後，請在終端機 print 出一份 Markdown 格式的「統計摘要表」（包含：最終 Loss、最終 PPL、平均每步耗時、預估總訓練時間等）。