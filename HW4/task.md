# 請扮演一位極度有耐心的 AI 程式導師。
# 我的目標是：重現 Andrej Karpathy 在 2026 年發布的「MicroGPT」專案。
# 我是一個完全不懂深奧數學的初學者，請把我當成智力偏低的笨蛋來教，註解要多白話就多白話。

# 請幫我用「純 Python (不依賴任何外部套件如 NumPy 或 PyTorch)」寫出一個單一檔案（約 200 行）的 GPT 模型。

# 程式碼必須包含以下 6 個核心部分，並且在每個部分加上「超級白話的繁體中文註解」：
# 1. Dataset (資料集)：寫一段可以自動下載 makemore names.txt 的程式碼，並讀取為清單。
# 2. Tokenizer (分詞器)：寫一個字元等級（character-level）的轉換器，把字母變成數字（包含一個特殊的 BOS 符號代表開始/結束）。
# 3. Autograd (自動微積分引擎)：實作一個簡單的 `Value` 類別，支援加、減、乘、次方、relu、exp、log，並能透過 .backward() 自動反向傳播算梯度（求導）。
# 4. Neural Network (神經網路架構)：實作類似 GPT-2 的架構，包含 linear (矩陣相乘)、softmax、rmsnorm，以及自注意力機制 (Self-Attention) 和 MLP。
# 5. Optimizer (優化器)：實作簡單的 Adam 優化器來更新參數。
# 6. Training & Inference Loop (訓練與推論迴圈)：寫一個可以反覆計算 Loss 並更新參數的迴圈，最後要能自動生成（推論）出幾個新的假名字印在畫面上。

# 額外要求：
# - 不要使用任何 object-oriented 以外的複雜設計，越直白越好。
# - 變數名稱盡量清楚，如果用縮寫（例如 lr, BOS, n_embd），務必在旁邊註解它到底是什麼意思。
# - 輸出請只給我完整的 Python 程式碼，讓我可以直接存成 microgpt.py 並執行。