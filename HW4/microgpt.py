#!/usr/bin/env python3
"""
================================================================================
microgpt.py - 用純 Python 寫的 GPT 模型（不用 NumPy 或 PyTorch）
================================================================================
這支程式是 Andrej Karpathy 在 2026 年發表的「MicroGPT」專案濃縮版。
我們把它改寫成超級白話的中文註解，讓任何人都能看懂。

這支程式從頭到尾包含：
  1. 資料集（自動下載人名檔案）
  2. 分詞器（把字母轉成數字）
  3. 自動微分引擎（自動算梯度）
  4. 神經網路（GPT-2 架構）
  5. 優化器（Adam）
  6. 訓練與推論迴圈

執行方式：python microgpt.py
================================================================================
"""

# 先把這支程式會用到的「工具」引用進來
# os：用來檢查檔案是否存在
# math：用來做數學運算（log, exp）
# random：用來產生隨機數
import os
import math
import random

# 把 random 的「亂數种子」固定，這樣每次執行的結果都會一樣（方便除錯）
random.seed(42)


# ================================================================================
# 第一部分：Dataset（資料集）
# ================================================================================
# 我們的「教材」是 Andrej Karpathy 的 makemore 專案提供的人名檔案（約 3.2 萬個名字）
# 下面的程式會自動去網路上下載這個檔案

# 先檢查一下，我們的電腦裡有沒有這個檔案（名叫 input.txt）
if not os.path.exists("input.txt"):
    # 如果沒有，就去網路上下載
    import urllib.request

    # 這是人名檔案的網址
    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")

# 把檔案裡的內容讀進來
# 我的解讀：打開 input.txt，一行一行讀取
#           每行去除前後空白後，加入 list
#           然後把所有行「洗牌」（打亂順序），這樣訓練更有效率
docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(docs)

# 印出來讓我們知道有多少筆資料
print(f"資料集（docs）裡總共有 {len(docs)} 個人名")


# ================================================================================
# 第二部分：Tokenizer（分詞器）
# ================================================================================
# Tokenizer 的工作：把「文字」轉成「數字」，電腦才能運算
# 我們用的是「字元等級」（character-level）的分詞器
# 例子：把 "alex" 轉成 [0, 1, 2, 3] 这样的数字序列

# 首先，找出資料集裡所有「不重複」的字母
# step 1: 把所有名字接在一起，变成一个大字符串
all_chars = "".join(docs)
# step 2: 把重複的字母去掉，只保留獨特的
unique_chars = sorted(set(all_chars))
# unique_chars 現在是 ['a', 'b', 'c', ...] 這樣的清單

# 這樣我們就知道每個字母對應到哪個數字了
# 比如說 'a' → 0, 'b' → 1, 'c' → 2, ...

# 我們再加一個「特殊符號」叫做 BOS（Begin Of Sequence）
# 這個符號用來表示「句子的開始」或「句子的結束」
# 為什麼要這個？因為我們要讓模型學習「什麼時候該開始說話」和「什麼時候該停止」
# 我們把 BOS 設定為「最大的那個數字」（也就是最後一個位置）
BOS = len(unique_chars)  # 這就是 BOS 的數字編號

# 總共有多少個「不一樣的符號」？
# 就是所有字母的数量 + BOS 这个特殊符号
vocab_size = len(unique_chars) + 1

# 印出來讓我們知道
print(f"詞彙表大小（vocab size）: {vocab_size} 個符號")


# ================================================================================
# 第三部分：Autograd（自動微分引擎）
# ================================================================================
# 這是整支程式最難懂的部分，但我會盡量解釋清楚！

# 先說明一下什麼是「梯度」（gradient）：
# 想像你在山上你要往山下走，你會往「最陡」的方向下去
# 那個「最陡的方向」就是梯度
# 我們用梯度來調整參數，讓 loss（損失）越來越小

# 這個 Value 類別是我們自己寫的「自動微分引擎」
# 支援：加(+)、減(-)、乘(*)、次方(**)、relu、exp、log
# 呼叫 .backward() 之後會自動算出所有參數的梯度


class Value:
    """
    Value類別：儲存一個「數字」，並且能自動算出它的梯度

    屬性：
    - data: 這個節點的实际數值（前向傳播算出來的）
    - grad: 這個節點對最終 loss 的導引程度（反向傳播算出來的）
    - _children: 這個節點的「父母」（依賴的節點）
    - _local_grads: 這個節點對每個「父母」的局部導數
    """

    # __slots__ 是 Python 的優化技術，讓記憶體用得更少
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        # 儲存這個節點的數值
        self.data = data
        # 初始化梯度為 0（等到 backward() 時才會正確計算）
        self.grad = 0
        # 儲存「依賴的節點」（父母）和「局部導數」
        self._children = children
        self._local_grads = local_grads

    # ========== 加法 (+) ==========
    def __add__(self, other):
        # 如果 other 不是 Value，幫它轉成 Value
        other = other if isinstance(other, Value) else Value(other)
        # 回傳：data = a + b, 局部導數都是 1（因為 d(a+b)/da = 1, d(a+b)/db = 1）
        return Value(self.data + other.data, (self, other), (1, 1))

    # ========== 乘法 (*) ==========
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 回傳：data = a * b, 局部導數是 b 和 a（乘法規則）
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    # ========== 次方 (**) ==========
    def __pow__(self, other):
        # 假設 y = x^n，則 dy/dx = n * x^(n-1)
        local_grad = other * self.data ** (other - 1)
        return Value(self.data**other, (self,), (local_grad,))

    # ========== 對數 log ==========
    def log(self):
        # dy/dx = 1/x（對數的導數公式）
        return Value(math.log(self.data), (self,), (1 / self.data,))

    # ========== 指數 exp ==========
    def exp(self):
        # dy/dx = e^x（指數的導數就是自己）
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    # ========== ReLU 啟動函數 ==========
    def relu(self):
        # relu(x) = max(0, x)
        # 導數：如果 x > 0 就是 1，否則是 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # ========== 負號 (-) ==========
    def __neg__(self):
        return self * -1

    # ========== 右乘（讓 2 * x 也能運作） ==========
    def __rmul__(self, other):
        return self * other

    # ========== 右加（讓 2 + x 也能運作） ==========
    def __radd__(self, other):
        return self + other

    # ========== 減法 (-) ==========
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # ========== 除法 (/) ==========
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    # ========== 反向傳播（算梯度）==========
    def backward(self):
        """
        反向傳播：從這個節點出發，沿著計算圖往回走
        自動把所有節點的梯度都算出來
        """
        # 建立「拓撲排序」（Topological Sort）
        # 簡單說：把計算圖弄成一個順序，確保「父母」在「小孩」之前
        # 這洋才能正確地從後往前算梯度
        topo = []
        visited = set()  # 記錄哪些節點已經處理過了

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)  # 遞迴處理所有「父母」
                topo.append(v)  # 把這個節點加到清單的尾巴

        build_topo(self)

        # 我們的目標函數（loss）對自己的導數是 1
        self.grad = 1

        # 從後往前，一個一個節點處理過去
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # 鏈鎖法則：child.grad += local_grad * v.grad
                child.grad += local_grad * v.grad


# ================================================================================
# 第四部分：Neural Network（神經網路架構）
# ================================================================================
# 我們要建立一個「GPT-2」架構的神經網路
# 不過我們做了一點簡化：
#   - LayerNorm → RMSNorm（更簡單）
#   - GeLU → ReLU（更簡單）
#   - 沒有用 bias（biases）
#   - 沒有效率最佳的 weight tying

# 設定一些「超參數」（hyperparameters）
n_layer = 1  # 網路有幾層（layer 的數量）
n_embd = 16  # embedding dimension：每個字母用 16 個數字來表示
block_size = 16  # 注意力視窗的大小（最長的名字是 15 個字母，我們給 16）
n_head = 4  # 注意頭（attention head）的數量
head_dim = n_embd // n_head  # 每個注意頭的維度


# 用來建立「矩陣」的輔助函數
# 會建立一個 list of list，每個元素都是一個 Value 物件
# 平均值是 0，標準差是 std（預設 0.08）
def matrix(nout, nin, std=0.08):
    """建立一個 nout x nin 的矩陣，每個元素都是隨機的 Value"""
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


# 建立「模型參數」（Model Parameters）
# 這些是我們的「知識」，會在訓練過程中一直被調整
state_dict = {
    # token embedding table：每個字母對應到一個 n_embd 維的向量
    "wte": matrix(vocab_size, n_embd),
    # position embedding table：每個位置對應到一個 n_embd 維的向量
    "wpe": matrix(block_size, n_embd),
    # 最後的「語言模型 head」（用來預測下一個字）
    "lm_head": matrix(vocab_size, n_embd),
}

# 對每一層，建立四個「權重矩陣」
# Query（查詢）、Key（鍵）、Value（值）、Output（輸出）
for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)  # 查詢矩陣
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)  # 鍵矩陣
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)  # 值矩陣
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)  # 輸出矩陣
    # MLP（多層感知器）的兩個矩陣
    # 這是 FFN（Feed-Forward Network）
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)  # first layer
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)  # second layer

# 把所有參數「展平成一維」，方便後續更新
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"模型總共有的參數數量（num params）: {len(params)}")


# ================================================================================
# 第五部分：定義網路的「零件」
# ================================================================================


def linear(x, w):
    """
    線性變換（矩陣相乘）
    輸入：x（n 維向量）, w（m x n 矩陣）
    輸出：y（m 維向量）
    y = x @ w^T 或者說 y_i = sum(x_j * w_ij)
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    """
    Softmax 函數：把「分數」轉成「機率」
    輸入：logits（還沒正規化的分數）
    輸出：機率分佈（所有值加起來 = 1）
    """
    # 先扣掉最大值（避免數字太大溢位）
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """
    RMSNorm（Root Mean Square Normalization）
    這是一種「標準化」的技術，可以讓訓練更穩定
    公式：y = x / sqrt(mean(x^2) + eps)
    """
    # 計算 mean of squares
    ms = sum(xi * xi for xi in x) / len(x)
    # 計算 scale
    scale = (ms + 1e-5) ** -0.5
    # 標準化
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    """
    GPT 模型的核心函數
    輸入：token_id（當前這個位置的字母編號）, pos_id（位置編號）, keys, values（KV cache）
    輸出：logits（對下一個字母的「預測分數」）
    """
    # 第一步：取得「Token Embedding」
    tok_emb = state_dict["wte"][token_id]

    # 第二步：取得「Position Embedding」
    pos_emb = state_dict["wpe"][pos_id]

    # 第三步：把兩個 embedding 加在一起
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    # 第四步：標準化（RMSNorm）
    x = rmsnorm(x)

    # 第五步：跑過每一層 Transformer
    for li in range(n_layer):
        # ----- 5.1 Multi-Head Attention（���頭注意力）-----
        # 先記住殘差，因為等等要加回去
        x_residual = x

        # 標準化
        x = rmsnorm(x)

        # 計算 Q（Query）、K（Key）、V（Value）
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])

        # 把 K 和 V 加入「快取」（KV Cache）
        # 這樣可以讓推論的時候更快
        keys[li].append(k)
        values[li].append(v)

        # 計算 Attention（注意力機制）
        x_attn = []
        for h in range(n_head):
            # 取出這個 head 的 Q、K、V
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]

            # 計算 Attention 的分數（Q 跟 K 的 dot product）
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            # 轉成機率
            attn_weights = softmax(attn_logits)

            # 用 Attention 的機率來「加權平均」V
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        # 輸出線性變換
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])

        # 加上殘差（residual connection）
        x = [a + b for a, b in zip(x, x_residual)]

        # ----- 5.2 MLP（多層感知器）-----
        x_residual = x
        x = rmsnorm(x)

        # 第一層： expanded dimension
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])

        # 啟動函數（ReLU）
        x = [xi.relu() for xi in x]

        # 第二層：壓回來
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])

        # 加上殘差
        x = [a + b for a, b in zip(x, x_residual)]

    # 最後：預測下一個字
    logits = linear(x, state_dict["lm_head"])
    return logits


# ================================================================================
# 第六部分：Optimizer（優化器）
# ================================================================================
# 我們用 Adam 優化器來更新參數
# Adam 是「Adaptive Moment Estimation」的縮寫
# 基本上是「動量」+「RMSProp」的組合

learning_rate = 0.01  # 學習率（learning rate）：每一步要走多遠
beta1 = 0.85  # Adam 的第一個超參數（動量）
beta2 = 0.99  # Adam 的第二個超參數（方差）
eps_adam = 1e-8  # 避免除零的小數

# Adam 的「 buffers」（緩衝區）
m = [0.0] * len(params)  # 第一動量（類似「動量」）
v = [0.0] * len(params)  # 第二動量（類似「方差」）


# ================================================================================
# 第七部分：Training Loop（訓練迴圈）
# ================================================================================
num_steps = 1000  # 要訓練多少步

# 正式開始訓練！
for step in range(num_steps):
    # 挑一個「教材」（一個名字）出來
    doc = docs[step % len(docs)]

    # 把名字轉成數字（tokenize）
    # 前面加 BOS，後面也加 BOS
    # 例子："alex" → [BOS, a, l, e, x, BOS]
    tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)  # 我們要預測 n 個字

    # 建立 KV Cache（空白的）
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    # 收集這一步的 loss
    losses = []

    # 對每個位置做「前向傳播」
    for pos_id in range(n):
        token_id = tokens[pos_id]  # 當前的字母
        target_id = tokens[pos_id + 1]  # 我們希望預測到的字母

        # 跑一次 GPT 模型
        logits = gpt(token_id, pos_id, keys, values)

        # 轉成機率
        probs = softmax(logits)

        # 計算 loss（就是 -log(prob)）
        # 我們希望正確答案的機率越高越好
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    # 計算平均 loss
    loss = (1 / n) * sum(losses)

    # 反向傳播（算梯度）
    loss.backward()

    # Adam 更新參數
    lr_t = learning_rate * (1 - step / num_steps)  # 學習率逐渐变小
    for i, p in enumerate(params):
        # 更新第一動量
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # 更新第二動量
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        # 修正偏差
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        # 更新參數
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        # 重置梯度（因為已經用完了）
        p.grad = 0

    # 印出進度
    print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end="\r")


# ================================================================================
# 第八部分：Inference Loop（推論迴圈）
# ================================================================================
# 訓練完了！現在讓模型自己「說話」

temperature = 0.5  # 溫度：控制「創造力」
# 低溫（< 1）：保守、確定性高
# 高溫（> 1）：創造力強、但可能乱说
print("\n--- 推論結果（模型「幻想」出來的新名字）---")

for sample_idx in range(20):
    # 重置 KV Cache
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    # 從 BOS 開始
    token_id = BOS
    sample = []

    # 一個一個字生出來
    for pos_id in range(block_size):
        # 跑模型
        logits = gpt(token_id, pos_id, keys, values)

        # 調整 temperature
        probs = softmax([l / temperature for l in logits])

        # 根據機率抽樣
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        # 如果抽到 BOS，代表已經結束了
        if token_id == BOS:
            break

        # 把字母加到清單
        sample.append(unique_chars[token_id])

    # 印出新名字
    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
