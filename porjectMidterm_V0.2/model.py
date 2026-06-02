"""
======================
model.py —— 模型工廠（v1.1 專業升級版）
只放 LSTM 語言模型，不使用 Transformer / Attention
就像一台「文字接龍機器」的大腦
======================
升級 v1.1：
- 加入 Weight Tying（權重共享）：讓 Embedding 層跟 Linear 輸出層共用同一組權重
  就像學生跟老師共用同一本課本，不用各自買一本，省空間又學得更好
"""

import torch
import torch.nn as nn


class MyLanguageModel(nn.Module):
    """
    語言模型神經網路：Embedding + LSTM + Linear。
    給它幾個詞，它會猜下一個詞是什麼。
    禁止使用 Transformer 或 Attention，所以用最經典的 LSTM。

    升級 v1.1：
    - 加入 Weight Tying（權重共享）機制
    - embed_size 現在預設等於 hidden_size（權重共享需要兩者大小一樣）
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=64, num_layers=2,
                 dropout=0.2, tie_weights=True):
        """
        初始化模型的三層結構。

        參數說明：
        - vocab_size：字典大小（總共有幾個不同的詞）
        - embed_size：每個詞用幾個數字來表示（預設 64）
        - hidden_size：LSTM 的記憶體容量（預設 64）
        - num_layers：LSTM 疊幾層（預設 2）
        - dropout：隨機遺忘比例（預設 0.2，防止死背訓練資料）
        - tie_weights：是否啟用權重共享（預設 True，開啟後 embed_size 會強制等於 hidden_size）
        """
        super().__init__()

        if tie_weights and embed_size != hidden_size:
            embed_size = hidden_size

        self.tie_weights = tie_weights
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Embedding 層：把「詞的編號」轉成「有意義的向量」
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Dropout 層：防止過度擬合的機制
        # 就像考試前老師故意遮住部分筆記，強迫學生真正理解而不是死背
        self.dropout = nn.Dropout(dropout)

        # LSTM 層：核心記憶單元，會一邊讀句子一邊記住前面看過的內容
        # num_layers=2 表示有兩層 LSTM 疊在一起，像兩層過濾網
        # batch_first=True 表示輸入形狀是 (樣本數, 序列長度, 向量大小)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Linear 層：把 LSTM 的記憶轉換成「每個詞的得分」
        # 啟用權重共享時 bias=False，因為 Embedding 層本身沒有 bias
        self.fc = nn.Linear(hidden_size, vocab_size, bias=not tie_weights)

        # Weight Tying：讓 Embedding 跟 Linear 共用同一組權重
        # 就像兩個人用同一本字典，不用一人一本，減少模型體積又學得更好
        if tie_weights:
            self.fc.weight = self.embedding.weight

    def forward(self, x, hidden_state=None):
        """
        讓資料流經三層網路，最後輸出預測結果。

        x：輸入的文字（形狀 = [樣本數, 序列長度]）
        hidden_state：LSTM 的上一個狀態（生成文字時會用到）
        """
        # 第一關：Embedding → [樣本數, 序列長度, embed_size]
        x = self.embedding(x)

        # 第二關：Dropout → 隨機遺忘部分資訊，防止死背
        x = self.dropout(x)

        # 第三關：LSTM → [樣本數, 序列長度, hidden_size]
        out, hidden_state = self.lstm(x, hidden_state)

        # 第四關：Linear → 取最後時間點，[樣本數, vocab_size]
        out = self.fc(out[:, -1, :])

        return out, hidden_state
