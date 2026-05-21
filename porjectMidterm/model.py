"""
======================
model.py —— 模型工廠（專業升級版）
只放 LSTM 語言模型，不使用 Transformer / Attention
就像一台「文字接龍機器」的大腦
======================
"""

import torch
import torch.nn as nn


class MyLanguageModel(nn.Module):
    """
    語言模型神經網路：Embedding + LSTM + Linear。
    給它幾個字，它會猜下一個字是什麼。
    禁止使用 Transformer 或 Attention，所以用最經典的 LSTM。

    升級：加入 Dropout 防止過度擬合，LSTM 改為 2 層。
    """

    def __init__(self, vocab_size, embed_size=32, hidden_size=64, num_layers=2, dropout=0.2):
        """
        初始化模型的三層結構。

        參數說明：
        - vocab_size：字典大小（總共有幾個不同的字）
        - embed_size：每個字用幾個數字來表示（預設 32）
        - hidden_size：LSTM 的記憶體容量（預設 64）
        - num_layers：LSTM 疊幾層（預設 2，Dropout 需要至少 2 層）
        - dropout：隨機遺忘比例（預設 0.2，防止死背訓練資料）
        """
        super().__init__()

        # Embedding 層：把「字的編號」轉成「有意義的向量」
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Dropout 層：防止過度擬合的機制
        # 就像考試前老師故意遮住部分筆記，強迫學生真正理解而不是死背
        # 訓練時隨機把 20% 的神經元關掉，讓模型學會不依賴單一線索
        self.dropout = nn.Dropout(dropout)

        # LSTM 層：核心記憶單元，會一邊讀句子一邊記住前面看過的內容
        # num_layers=2 表示有兩層 LSTM 疊在一起，像兩層過濾網
        # dropout=0.2 表示層與層之間隨機遺忘 20%，防止過度擬合
        # batch_first=True 表示輸入形狀是 (樣本數, 序列長度, 向量大小)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Linear 層：把 LSTM 的記憶轉換成「每個字的得分」
        self.fc = nn.Linear(hidden_size, vocab_size)

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
