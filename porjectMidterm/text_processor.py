"""
======================
text_processor.py —— 資料處理小幫手（專業升級版）
負責：建立字典、文字變數字、準備訓練樣本、外部資料讀取、DataLoader 批次推車
就像一個「文字加工廠」，把中文字變成模型看得懂的數字
======================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


# ---------- 備用假資料（約 50 字的中文小故事） ----------
FALLBACK_STORY = "從前有一個小男孩住在森林裡的小木屋，他每天都會去河邊釣魚。有一天他發現了一條金色的魚，那條魚居然開口說話了！"


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
                print(f"[✓] 成功讀取外部資料：{filepath}（共 {len(text)} 字）")
                return text
        except Exception as e:
            print(f"[!] 讀取 {filepath} 失敗：{e}")
    print(f"[!] 找不到 {filepath}，使用備用假故事")
    return FALLBACK_STORY


def build_vocab(text):
    """
    建立「字→數字」跟「數字→字」兩本字典。

    參數：
    - text：原始中文字串

    回傳：
    - char_to_id：字→編號 的字典
    - id_to_char：編號→字 的字典
    - vocab_size：字典大小（總共有幾個不同的字）
    """
    unique_chars = sorted(list(set(text)))
    char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
    id_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    vocab_size = len(unique_chars)
    return char_to_id, id_to_char, vocab_size


def text_to_ids(text, char_to_id):
    """
    把一串中文字轉成一串數字編號。

    參數：
    - text：原始中文字串
    - char_to_id：字→編號 的字典

    回傳：
    - 數字串列，例如 "你好" → [5, 3]
    """
    return [char_to_id[ch] for ch in text]


def prepare_training_data(ids_sequence, sequence_length=8):
    """
    準備訓練用的輸入樣本跟正確答案。

    參數：
    - ids_sequence：整篇故事的數字串列
    - sequence_length：用前面幾個字來預測下一個字（預設 8）

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
