"""
======================
main.py —— 主程式與老闆（專業升級版）
負責把 text_processor 跟 model 兩個小幫手 import 進來，
然後執行「訓練迴圈」跟最後的「文字生成」展示

升級：存檔讀檔機制、DataLoader 批次訓練、Temperature 採樣
======================
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
)
from model import MyLanguageModel


# =====================================================================
# 設定區：超參數與檔案路徑
# =====================================================================

CHECKPOINT_PATH = "model_checkpoint.pt"  # 模型權重存檔路徑
BATCH_SIZE = 64                         # 每次訓練搬幾筆資料（DataLoader 推車容量）
TOTAL_EPOCHS = 150                      # 總共訓練幾輪
LEARNING_RATE = 0.01                    # 學習率：每次調整參數的幅度
SEQUENCE_LENGTH = 8                     # 用前面幾個字預測下一個字
TEMPERATURE = 0.8                       # 文字生成溫度：越低越保守，越高越有創意


# =====================================================================
# 第一步：文字預處理 — 嘗試讀外部 txt，找不到就用備用故事
# =====================================================================

story = load_external_text("dataset.txt")

char_to_id, id_to_char, vocab_size = build_vocab(story)
print(f"字典大小（總共有幾個不同的字）: {vocab_size}")
print()

ids_sequence = text_to_ids(story, char_to_id)
inputs, targets = prepare_training_data(ids_sequence, sequence_length=SEQUENCE_LENGTH)

# 建立 DataLoader：批次推車模式
# 原本是一次把所有資料塞給模型，現在改成每次只搬 batch_size 筆
# 就像原本一次搬 100 箱貨物，現在改成每次搬 4 箱，搬 25 次
train_loader = create_dataloader(inputs, targets, batch_size=BATCH_SIZE, shuffle=True)

print(f"總共有 {len(inputs)} 組訓練樣本")
print(f"批次大小：{BATCH_SIZE}，每輪需要 {len(train_loader)} 個批次")
print()

# =====================================================================
# 第二步：建立模型
# =====================================================================

model = MyLanguageModel(vocab_size, embed_size=32, hidden_size=64, num_layers=2, dropout=0.2)

# =====================================================================
# 第三步：訓練迴圈 — 讓模型變聰明（批次訓練 + 存檔機制）
# =====================================================================

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("開始訓練模型...")
print("=" * 50)

for epoch in range(TOTAL_EPOCHS):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    # 用 DataLoader 逐批次訓練，每次只搬 batch_size 筆資料
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        predictions, _ = model(batch_inputs)
        loss = loss_function(predictions, batch_targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    # 計算該輪的平均 Loss
    avg_loss = epoch_loss / batch_count

    # 每 15 個 Epoch 印一次 Loss
    if (epoch + 1) % 15 == 0:
        print(f"第 {epoch+1:3d} / {TOTAL_EPOCHS} 回合 | 平均 Loss: {avg_loss:.4f}")

print("=" * 50)
print("訓練完成！")
print()

# =====================================================================
# 存檔機制：把訓練好的模型權重存成檔案
# 就像把考試前複習好的筆記存成 PDF，下次可以直接看，不用重新準備
# =====================================================================

torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"[✓] 模型權重已儲存至：{CHECKPOINT_PATH}")


# =====================================================================
# 讀檔機制：載入已儲存的模型權重
# 如果已經有存檔，就直接讀取，不用重新訓練
# =====================================================================

def load_model(model, checkpoint_path):
    """
    從檔案讀取模型權重。

    就像去書櫃拿之前存好的筆記本，直接接著上次的進度繼續用。

    參數：
    - model：要建立權重的模型物件
    - checkpoint_path：權重檔案路徑

    回傳：
    - 載入權重後的模型
    """
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"[✓] 成功載入模型權重：{checkpoint_path}")
    else:
        print(f"[!] 找不到權重檔案：{checkpoint_path}")
    return model


# 示範讀檔：建立一個新模型並載入權重
print("\n--- 讀檔示範 ---")
new_model = MyLanguageModel(vocab_size, embed_size=32, hidden_size=64, num_layers=2, dropout=0.2)
new_model = load_model(new_model, CHECKPOINT_PATH)


# =====================================================================
# 第四步：文字生成 — 給開頭字，自動接龍（加入 Temperature 採樣）
# =====================================================================

def generate_text(model, start_char, max_length=15, temperature=0.8):
    """
    文字接龍函數。
    給模型一個「開頭的字」，它會自動生出後面的句子。

    Temperature（溫度）控制生成文字的「隨機度與創意度」：
    - temperature < 1（例如 0.5）：模型說話更保守、更可預測
      就像冬天，大家擠在一起，只選機率最高的字
    - temperature = 1（例如 1.0）：按照原始機率抽樣
      就像春天，一切正常
    - temperature > 1（例如 1.5）：模型說話更有創意、更隨機
      就像夏天，大家散開，機率低的字也有機會被選到

    參數：
    - start_char：開頭的字（字串，例如 "從"）
    - max_length：要生成幾個字（不包含開頭字）
    - temperature：溫度參數（預設 0.8，稍微保守但仍有變化）

    回傳：
    - 完整的生成句子（字串）
    """
    model.eval()
    generated_chars = [start_char]
    current_input = torch.tensor([[char_to_id[start_char]]])
    current_hidden = None

    with torch.no_grad():
        for _ in range(max_length):
            output, current_hidden = model(current_input, current_hidden)

            # Temperature 採樣：把 logits 除以溫度，控制隨機度
            # 溫度越低，機率分佈越尖銳（高機率的字更突出）
            # 溫度越高，機率分佈越平坦（每個字都有機會）
            scaled_logits = output / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)

            # 根據機率抽籤：機率越高的字越容易被抽到
            next_id = torch.multinomial(probabilities, 1).item()
            next_char = id_to_char[next_id]
            generated_chars.append(next_char)
            current_input = torch.tensor([[next_id]])

    return "".join(generated_chars)


# ========== 展示生成結果 ==========
print("\n文字生成展示")
print("=" * 50)

# 用不同的溫度來生成，看看效果差異
temperatures = [0.5, 0.8, 1.2]
starting_words = ["從", "他", "那", "有"]

for temp in temperatures:
    print(f"\n🌡️ Temperature = {temp}")
    print("-" * 40)
    for word in starting_words:
        result = generate_text(new_model, word, max_length=15, temperature=temp)
        print(f"開頭「{word}」 → {result}")

print("\n" + "=" * 50)
