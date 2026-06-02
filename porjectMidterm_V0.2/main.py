"""
======================
main.py —— 主程式與老闆（v1.1 專業升級版）
負責把 text_processor 跟 model 兩個小幫手 import 進來，
然後執行「訓練迴圈」跟最後的「文字生成」展示
======================
升級 v1.1：
- 自動偵測 GPU/CUDA / MPS / CPU，不再只綁死在 CPU
- Top-k 採樣：生成文字時只從最熱門的前 k 個詞中抽樣，避免火星文
- 配合 jieba 分詞跟 Weight Tying 新架構
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from text_processor import (
    load_external_text,
    build_vocab,
    text_to_ids,
    prepare_training_data,
    create_dataloader,
    auto_device,
)
from model import MyLanguageModel


# =====================================================================
# 設定區：超參數與檔案路徑
# =====================================================================

CHECKPOINT_PATH = "model_checkpoint.pt"  # 模型權重存檔路徑
BATCH_SIZE = 64                         # 每次訓練搬幾筆資料（DataLoader 推車容量）
TOTAL_EPOCHS = 1500                      # 總共訓練幾輪
LEARNING_RATE = 0.0001                    # 學習率：每次調整參數的幅度
SEQUENCE_LENGTH = 10                     # 用前面幾個詞預測下一個詞
TEMPERATURE = 0.7                       # 文字生成溫度：越低越保守，越高越有創意
TOP_K = 5                               # Top-k 採樣：只從前 k 個熱門詞中抽樣
PATIENCE = 100                          # 提早停止機制：容忍幾回合 Loss 沒下降


# =====================================================================
# 第一步：文字預處理 — 嘗試讀外部 txt，找不到就用備用故事
# =====================================================================

story = load_external_text("dataset.txt")

word_to_id, id_to_word, vocab_size = build_vocab(story)
print(f"字典大小（總共有幾個不同的詞）: {vocab_size}")
print()

ids_sequence = text_to_ids(story, word_to_id)
inputs, targets = prepare_training_data(ids_sequence, sequence_length=SEQUENCE_LENGTH)

# 建立 DataLoader：批次推車模式
train_loader = create_dataloader(inputs, targets, batch_size=BATCH_SIZE, shuffle=True)

print(f"總共有 {len(inputs)} 組訓練樣本")
print(f"批次大小：{BATCH_SIZE}，每輪需要 {len(train_loader)} 個批次")
print()

# =====================================================================
# 第二步：建立模型 + 自動偵測硬體（GPU / MPS / CPU）
# =====================================================================

device = auto_device()
print(f"使用的運算裝置：{device}")
print()

model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, num_layers=2, dropout=0.2)
model = model.to(device)

# =====================================================================
# 第三步：訓練迴圈 — 讓模型變聰明（批次訓練 + 存檔機制）
# =====================================================================

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("開始訓練模型...")
print("=" * 50)

best_loss = float('inf')
patience_counter = 0

for epoch in range(TOTAL_EPOCHS):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for batch_inputs, batch_targets in train_loader:
        # 把資料搬到 GPU 或 CPU 上
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        predictions, _ = model(batch_inputs)
        loss = loss_function(predictions, batch_targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count

    if (epoch + 1) % 15 == 0:
        print(f"第 {epoch+1:3d} / {TOTAL_EPOCHS} 回合 | 平均 Loss: {avg_loss:.4f}")

    # Early Stopping 與儲存最佳模型機制
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # 只要有進步，就先存檔
        torch.save(model.state_dict(), CHECKPOINT_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n[!] Loss 已經連續 {PATIENCE} 回合沒有下降，提早結束訓練！(在第 {epoch+1} 回合)")
            break

print("=" * 50)
print("訓練完成！")
print()

# =====================================================================
# 存檔機制
# =====================================================================

print(f"[v] 最佳模型權重已確認儲存至：{CHECKPOINT_PATH} (最低 Loss: {best_loss:.4f})")


# =====================================================================
# 讀檔機制
# =====================================================================

def load_model(model, checkpoint_path):
    """
    從檔案讀取模型權重。
    就像去書櫃拿之前存好的筆記本，直接接著上次的進度繼續用。
    """
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"[v] 成功載入模型權重：{checkpoint_path}")
    else:
        print(f"[!] 找不到權重檔案：{checkpoint_path}")
    return model


print("\n--- 讀檔示範 ---")
new_model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, num_layers=2, dropout=0.2)
new_model = new_model.to(device)
new_model = load_model(new_model, CHECKPOINT_PATH)


# =====================================================================
# 第四步：文字生成 — Top-k 採樣 + Temperature（給開頭詞，自動接龍）
# =====================================================================

def top_k_filter(logits, k=5):
    """
    Top-k 過濾：只保留機率最高的前 k 個選項，其他的全部變成 -inf（不可能被抽到）。
    就像在超市選購時，先從貨架上挑出最熱門的前 5 種商品，
    再從這 5 種裡面慢慢選，不會浪費時間看整排貨架。

    參數：
    - logits：模型輸出的原始分數（形狀 = [vocab_size]）
    - k：要保留的熱門選項數量（預設 5）

    回傳：
    - 過濾後的 logits（只有前 k 個保留，其餘為 -inf）
    """
    if k <= 0:
        return logits
    values, indices = torch.topk(logits, k)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(0, indices, values)
    return mask


def generate_text(model, start_word, word_to_id, id_to_word,
                  max_length=15, temperature=0.8, top_k=5):
    """
    文字接龍函數（加入 Top-k 採樣）。
    給模型一個「開頭的詞」，它會自動生出後面的句子。

    Temperature（溫度）控制生成文字的「隨機度與創意度」：
    - temperature < 1（例如 0.5）：模型說話更保守、更可預測
    - temperature = 1（例如 1.0）：按照原始機率抽樣
    - temperature > 1（例如 1.5）：模型說話更有創意、更隨機

    Top-k 採樣：只從機率最高的前 k 個詞中抽樣，避免選到奇怪的字。

    參數：
    - start_word：開頭的詞（字串，例如 "從"）
    - word_to_id：詞→編號 字典
    - id_to_word：編號→詞 字典
    - max_length：要生成幾個詞（不包含開頭詞）
    - temperature：溫度參數（預設 0.8）
    - top_k：只從前 k 個熱門詞中選（預設 5，0 表示關閉）

    回傳：
    - 完整的生成句子（字串）
    """
    model.eval()

    start_id = word_to_id.get(start_word, 0)
    generated_words = [id_to_word[start_id]]

    current_input = torch.tensor([[start_id]], device=next(model.parameters()).device)
    current_hidden = None

    with torch.no_grad():
        for _ in range(max_length):
            output, current_hidden = model(current_input, current_hidden)

            # 把形狀從 [1, vocab_size] 壓成 [vocab_size] 方便處理
            logits = output.squeeze(0).squeeze(0)

            # 第一步：Top-k 過濾 — 只留前 k 個熱門選項
            if top_k > 0:
                logits = top_k_filter(logits, k=top_k)

            # 第二步：Temperature 縮放 — 控制隨機度
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)

            # 第三步：根據機率抽籤
            next_id = torch.multinomial(probabilities, 1).item()
            next_word = id_to_word[next_id]
            generated_words.append(next_word)
            current_input = torch.tensor([[next_id]], device=current_input.device)

    return "".join(generated_words)


# ========== 展示生成結果 ==========
print("\n文字生成展示（Top-k = 5 + Temperature）")
print("=" * 50)

temperatures = [0.5, 0.8, 1.2]
starting_words = ["從", "他", "那", "有"]

for temp in temperatures:
    print(f"\n溫度 Temperature = {temp}")
    print("-" * 40)
    for word in starting_words:
        result = generate_text(new_model, word, word_to_id, id_to_word,
                               max_length=15, temperature=temp, top_k=TOP_K)
        print(f"開頭「{word}」 -> {result}")

print("\n" + "=" * 50)
