"""
======================
text_processor.py —— 資料處理小幫手（v1.1 專業升級版）
負責：jieba 分詞、建立詞典、文字轉數字、準備訓練樣本、DataLoader
就像一個「文字加工廠」，把中文句子變成模型看得懂的數字
======================
升級 v1.1：
- 從「字層級」升級為「詞層級」，使用 jieba 分詞（像把句子切成一個個詞）
- 加入 <UNK> 未知詞標記（像字典裡查不到的詞就畫個問號）
- 遇到沒看過的詞不會當機，自動用 <UNK> 代替
"""

import os
from collections import Counter
import jieba
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- 特殊標記 ----------
UNK_TOKEN = "<UNK>"       # 未知詞標記：字典裡找不到的詞就用這個
UNK_ID = 0                # 未知詞的編號固定為 0


# ---------- 備用假資料（約 50 字的中文小故事） ----------
FALLBACK_STORY = "從前有一個小男孩住在森林裡的小木屋，他每天都會去河邊釣魚。有一天他發現了一條金色的魚，那條魚居然開口說話了！"


def auto_device():
    """
    自動偵測可用硬體，依序檢查：CUDA → MPS → CPU。
    就像手機自動切換 Wi-Fi 和行動網路：
    有 Wi-Fi（CUDA）就用 Wi-Fi，
    沒 Wi-Fi 但有藍牙（MPS）就用藍牙，
    都沒有就用行動網路（CPU）。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_external_text(filepath="dataset.txt"):
    """
    嘗試從外部 .txt 檔案讀取文字。
    如果檔案不存在或讀取失敗，就使用備用假故事。

    就像去圖書館借書：借得到就用，借不到就用自己的筆記本。

    參數：
    - filepath：外部文本檔案路徑

    回傳：
    - 讀取到的文字串列（字串）
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if len(text) > 0:
                print(f"[v] 成功讀取外部資料：{filepath}（共 {len(text)} 字）")
                return text
        except Exception as e:
            print(f"[!] 讀取 {filepath} 失敗：{e}")
    print(f"[!] 找不到 {filepath}，使用備用假故事")
    return FALLBACK_STORY


def tokenize(text):
    """
    用 jieba 把句子切成一塊一塊的「詞」。
    就像把「我愛吃蘋果」切成「我 / 愛吃 / 蘋果」。

    參數：
    - text：原始中文字串

    回傳：
    - 詞語串列，例如 ["我", "愛吃", "蘋果"]
    """
    return list(jieba.lcut(text))


def build_vocab(text, min_freq=1):
    """
    建立「詞→數字」跟「數字→詞」兩本字典（詞層級）。

    字典裡會自動加入 <UNK>（未知詞）標記，編號固定為 0。
    就像在字典的第一頁先放一個「問號頁」，查不到的詞都歸到這裡。

    參數：
    - text：原始中文字串
    - min_freq：最低出現次數，低於此次數的詞會被丟棄並映射到 <UNK>（預設 1 = 保留全部）

    回傳：
    - word_to_id：詞→編號 的字典
    - id_to_word：編號→詞 的字典
    - vocab_size：字典大小（總共有幾個不同的詞，含 <UNK>）
    """
    words = tokenize(text)
    freq = Counter(words)
    kept_words = sorted([w for w in freq if freq[w] >= min_freq])
    word_to_id = {word: i + 1 for i, word in enumerate(kept_words)}
    word_to_id[UNK_TOKEN] = UNK_ID
    id_to_word = {i + 1: word for i, word in enumerate(kept_words)}
    id_to_word[UNK_ID] = UNK_TOKEN
    vocab_size = len(kept_words) + 1
    if min_freq > 1:
        dropped = sum(1 for v in freq.values() if v < min_freq)
        print(f"[v] 低頻詞過濾: min_freq={min_freq}，捨棄 {dropped} 個稀有詞，保留 {len(kept_words)} 個詞")
    return word_to_id, id_to_word, vocab_size


def text_to_ids(text, word_to_id):
    """
    把一串中文句子轉成一串數字編號（詞層級）。
    如果有沒看過的詞，就用 <UNK>（編號 0）代替，不會當機。

    就像老師發考卷，看不懂的單字先用紅筆圈起來標記「待查」。

    參數：
    - text：原始中文字串
    - word_to_id：詞→編號 的字典

    回傳：
    - 數字串列，例如 "我愛吃蘋果" → [2, 5, 8]（假設的編號）
    """
    words = tokenize(text)
    return [word_to_id.get(word, UNK_ID) for word in words]


def prepare_training_data(ids_sequence, sequence_length=8):
    """
    準備訓練用的輸入樣本跟正確答案。

    參數：
    - ids_sequence：整篇故事的數字串列
    - sequence_length：用前面幾個詞來預測下一個詞（預設 8）

    回傳：
    - inputs：輸入樣本（二維串列）
    - targets：正確答案（串列）
    """
    inputs = []
    targets = []
    for i in range(len(ids_sequence) - sequence_length):
        inputs.append(ids_sequence[i : i + sequence_length])
        targets.append(ids_sequence[i + sequence_length])
    return inputs, targets


class TextDataset(Dataset):
    """
    自訂 PyTorch Dataset，讓 DataLoader 可以批次載入資料。

    就像一台「手推車」：每次只搬 batch_size 個樣本給模型，
    不用一次把所有資料塞進去，避免記憶體爆掉。

    用法：
    >>> dataset = TextDataset(inputs, targets)
    >>> loader = DataLoader(dataset, batch_size=4, shuffle=True)
    >>> for batch_inputs, batch_targets in loader:
    ...     # 每次拿到 batch_size 筆資料
    """

    def __init__(self, inputs, targets):
        """
        參數：
        - inputs：輸入樣本（二維串列或 Tensor）
        - targets：正確答案（串列或 Tensor）
        """
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        """回傳總共有幾筆資料"""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        根據索引回傳一筆資料。
        DataLoader 會自動呼叫這個函數來組裝一個 batch。
        """
        return self.inputs[idx], self.targets[idx]


def create_dataloader(inputs, targets, batch_size=4, shuffle=True):
    """
    建立 DataLoader 的快捷函數。

    參數：
    - inputs：輸入樣本
    - targets：正確答案
    - batch_size：每次搬幾筆（預設 4）
    - shuffle：是否打亂順序（預設 True，避免模型死記順序）

    回傳：
    - DataLoader 物件，可以用 for 迴圈逐批次取出資料
    """
    dataset = TextDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
