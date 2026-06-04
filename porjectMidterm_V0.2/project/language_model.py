"""
======================================
    語言模型（Language Model）
    使用 LSTM 做中文文字生成
    禁止使用 Transformer / Attention
======================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 第一部分：假資料（自己編一個約 50 字的中文小故事）
# =====================================================================

story = "從前有一個小男孩住在森林裡的小木屋，他每天都會去河邊釣魚。有一天他發現了一條金色的魚，那條魚居然開口說話了！"

# =====================================================================
# 第二部分：文字預處理（把中文字變成數字字典）
# =====================================================================

# set(story) 會把故事中「不重複」的字全部挑出來
# sorted(...) 把這些字排序，確保每次執行順序都一樣
# list(...) 轉成串列（Python 的清單）
unique_chars = sorted(list(set(story)))

# 建立「字→數字」的字典：把每個中文字對應到一個獨一無二的編號
# enumerate 會給每個字一個號碼（0, 1, 2, ...）
char_to_id = {ch: i for i, ch in enumerate(unique_chars)}

# 建立「數字→字」的字典：反過來，看到數字就知道是哪個字
id_to_char = {i: ch for i, ch in enumerate(unique_chars)}

# 字典的大小，也就是總共有多少個不同的字
# 模型最後要從這些字裡面「選一個」當作預測結果
vocab_size = len(unique_chars)

print(f"字典大小（總共有幾個不同的字）: {vocab_size}")
print(f"字典內容: {char_to_id}")
print()

# 把整個故事從「一串文字」轉成「一串數字」
# 例如 "你好" 變成 [5, 3]（假設 5=你, 3=好）
ids_sequence = [char_to_id[ch] for ch in story]

# 決定用前面幾個字來預測下一個字（就像給你一句話的前 8 個字，猜第 9 個字）
sequence_length = 8

# 準備訓練資料：
# - inputs：拿連續幾個字當作「輸入樣本」
# - targets：這幾個字的下一個字當作「正確答案」
inputs = []
targets = []
for i in range(len(ids_sequence) - sequence_length):
    # 從位置 i 開始，連續取 sequence_length 個字
    inputs.append(ids_sequence[i : i + sequence_length])
    # 這 sequence_length 個字的「下一個字」就是正確答案
    targets.append(ids_sequence[i + sequence_length])

# 把 Python 串列轉成 PyTorch 的張量（Tensor）
# 張量就像一個「超強陣列」，GPU 可以加速計算
inputs_tensor = torch.tensor(inputs)
targets_tensor = torch.tensor(targets)

print(f"總共有 {len(inputs)} 組訓練樣本")
print(f"輸入形狀（樣本數 × 序列長度）: {inputs_tensor.shape}")
print(f"目標形狀（樣本數）: {targets_tensor.shape}")
print()

# =====================================================================
# 第三部分：LSTM 模型架構（Embedding + LSTM + Linear）
# =====================================================================

class MyLanguageModel(nn.Module):
    """
    這是一個「語言模型」的神經網路。
    可以把它想像成一台「文字接龍機器」——給它幾個字，它會猜下一個字是什麼。
    禁止使用 Transformer 或 Attention，所以我們用最經典的 LSTM。
    """

    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        初始化模型的三層結構。

        參數說明：
        - vocab_size：字典大小（總共有幾個不同的字）
        - embed_size：每個字用幾個數字來表示（越大表示越精細，但也越耗記憶體）
        - hidden_size：LSTM 的記憶體容量（越大記得越多，但也越慢）
        """
        super().__init__()

        # Embedding 層：把「字的編號」轉成「有意義的向量」
        # 就像查辭典：輸入「3號字」→ 輸出一組代表這個字的數字陣列
        # 模型會自己學習這些數字陣列應該長怎樣
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM 層：核心記憶單元
        # 它像一個「有記憶的筆記本」，會一邊讀句子一邊記住前面看過的內容
        # batch_first=True 表示輸入形狀是 (樣本數, 序列長度, 向量大小)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # Linear 層（全連接層）：把 LSTM 的記憶轉換成「每個字的得分」
        # 就像考試成績單：每個字都得到一個分數，分數越高的字越可能是答案
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        """
        讓資料流經三層網路，最後輸出預測結果。

        x：輸入的文字（形狀 = [樣本數, 序列長度]）
        hidden_state：LSTM 的上一個狀態（生成文字時會用到，訓練時讓它自動初始化）
        """
        # 第一關：Embedding 層
        # 把 [樣本數, 序列長度] → [樣本數, 序列長度, embed_size]
        # 就像把「字的編號」換成「字的意義向量」
        x = self.embedding(x)

        # 第二關：LSTM 層
        # [樣本數, 序列長度, embed_size] → [樣本數, 序列長度, hidden_size]
        # LSTM 會讀完整段話，記住前後文的關係
        # out 是每個時間點的輸出，hidden_state 是記憶狀態
        out, hidden_state = self.lstm(x, hidden_state)

        # 第三關：Linear 層
        # 取「最後一個時間點」的輸出做預測
        # out[:, -1, :] 意思是「拿所有樣本的最後一個字的那組數字」
        # [樣本數, hidden_size] → [樣本數, vocab_size]
        # 最後得到每個字的得分，得分最高的就是模型認為的下一個字
        out = self.fc(out[:, -1, :])

        return out, hidden_state


# 建立模型實體
#
# 可以調的參數（超參數 Hyperparameters）：
# - embed_size = 32：每個字用 32 個數字來表示
# - hidden_size = 64：LSTM 有 64 個單元的記憶容量
model = MyLanguageModel(vocab_size, embed_size=32, hidden_size=64)

# =====================================================================
# 第四部分：訓練迴圈（讓模型變聰明）
# =====================================================================

# CrossEntropyLoss = 交叉熵損失函數
# 就是用來計算「模型猜的答案」跟「正確答案」差多少
# 差越多，Loss 數字就越大
loss_function = nn.CrossEntropyLoss()

# Adam 優化器：負責調整模型的參數，讓 Loss 越來越小
# 就像老師調整學生的答題策略，讓每次考試分數越來越高
# lr = learning_rate（學習率）= 0.01，表示每次調整的幅度
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 總共要訓練幾個回合（Epoch）
# 一個 Epoch 代表模型把整份訓練資料看過一遍
total_epochs = 150

print("開始訓練模型...")
print("=" * 50)

for epoch in range(total_epochs):
    # 設定為訓練模式（讓模型啟用訓練相關功能，例如 Dropout）
    model.train()

    # 把優化器裡面累積的梯度清空
    # 梯度就像「調整方向」，不清空會跟下一次的混在一起
    optimizer.zero_grad()

    # 把訓練資料丟進模型，得到預測結果
    # predictions 的形狀 = [樣本數, 字典大小]
    predictions, _ = model(inputs_tensor)

    # 計算 Loss：比較「預測結果」跟「正確答案」差多少
    loss = loss_function(predictions, targets_tensor)

    # 反向傳播：根據 Loss 計算每個參數的梯度
    # 就像分析「這次哪裡猜錯了，該往哪個方向調整」
    loss.backward()

    # 優化器根據梯度更新模型參數
    # 實際執行「調整參數」這一步
    optimizer.step()

    # 每 15 個 Epoch 印一次 Loss，讓你看見模型在進步
    if (epoch + 1) % 15 == 0:
        print(f"第 {epoch+1:3d} / {total_epochs} 回合 | Loss: {loss.item():.4f}")

print("=" * 50)
print("訓練完成！")
print()

# =====================================================================
# 第五部分：文字生成（給開頭字，自動接龍）
# =====================================================================

def generate_text(model, start_char, max_length=15):
    """
    文字接龍函數。
    給模型一個「開頭的字」，它會自動生出後面的句子。

    參數：
    - start_char：開頭的字（字串，例如 "從"）
    - max_length：要生成幾個字（不包含開頭字）

    回傳：
    - 完整的生成句子（字串）
    """
    # 設定為評估模式（關閉訓練用的 Dropout 等功能）
    model.eval()

    # 存放生成結果，先放入開頭字
    generated_chars = [start_char]

    # 把開頭字轉成對應的數字編號
    # [[ ]] 包兩層是因為模型要接收 [樣本數, 序列長度] 的形狀
    # 這裡樣本數=1，序列長度=1
    current_input = torch.tensor([[char_to_id[start_char]]])

    # 初始的 LSTM 記憶狀態設為 None，讓它自動從零開始
    current_hidden = None

    # torch.no_grad()：告訴 PyTorch 不用計算梯度
    # 因為生成時不需要訓練，關掉梯度可以省記憶體跟加快速度
    with torch.no_grad():
        for _ in range(max_length):
            # 把當前字丟進模型，預測下一個字
            output, current_hidden = model(current_input, current_hidden)

            # 把模型輸出的得分轉成機率（所有字的機率加起來 = 1）
            # softmax 就像把得分換算成「中獎機率」
            probabilities = torch.softmax(output, dim=-1)

            # 根據機率來抽籤：機率越高的字越容易被抽到
            # 這樣不會每次都選一樣的字，讓生成結果更有變化
            next_id = torch.multinomial(probabilities, 1).item()

            # 把數字編號轉回中文字
            next_char = id_to_char[next_id]

            # 把生成的字加到結果清單中
            generated_chars.append(next_char)

            # 把剛產生的字當作下一次的輸入
            current_input = torch.tensor([[next_id]])

    # 把清單中的所有字串接起來，回傳完整的句子
    return "".join(generated_chars)


# ========== 展示生成結果 ==========
print("文字生成展示")
print("=" * 50)

# 用不同的開頭字來生成，看看模型學得怎樣
starting_words = ["從", "他", "那", "有"]
for word in starting_words:
    result = generate_text(model, word, max_length=15)
    print(f"開頭「{word}」 → {result}")

print("=" * 50)
