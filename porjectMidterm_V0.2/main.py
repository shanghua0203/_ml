"""
main.py —— 主程式與老闆（v1.2 專業升級版）
負責訓練迴圈 + 驗證 + 文字生成展示
"""

import os
import csv
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from text_processor import (
    load_external_text,
    build_vocab,
    text_to_ids,
    prepare_training_data,
    create_dataloader,
    TextDataset,
    auto_device,
)
from model import MyLanguageModel


CHECKPOINT_PATH = "model_checkpoint.pt"
BATCH_SIZE = 64
TOTAL_EPOCHS = 500
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 15
TEMPERATURE = 0.75
TOP_K = 15
PATIENCE = 200
VAL_SPLIT = 0.1
CLIP_GRAD_NORM = 1.0


story = load_external_text("dataset.txt")

word_to_id, id_to_word, vocab_size = build_vocab(story)
print(f"字典大小: {vocab_size}\n")

ids_sequence = text_to_ids(story, word_to_id)
inputs, targets = prepare_training_data(ids_sequence, sequence_length=SEQUENCE_LENGTH)

dataset = TextDataset(inputs, targets)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

print(f"總訓練樣本: {len(train_dataset)}，驗證樣本: {len(val_dataset)}")
print(f"批次大小: {BATCH_SIZE}，每輪訓練 {len(train_loader)} 批次，驗證 {len(val_loader)} 批次\n")

device = auto_device()
print(f"使用的運算裝置: {device}\n")

model = MyLanguageModel(vocab_size, embed_size=256, hidden_size=256,
                         num_layers=2, dropout=0.1)
model = model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

log_txt_path = "training_log.txt"
log_csv_path = "training_log.csv"

with open(log_txt_path, "w", encoding="utf-8") as f:
    f.write("LSTM 語言模型訓練日誌\n")
    f.write(f"啟動時間: {datetime.datetime.now()}\n")
    f.write(f"vocab_size={vocab_size}, embed_size=256, hidden_size=256, "
            f"num_layers=2, dropout=0.1\n")
    f.write(f"batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
            f"sequence_length={SEQUENCE_LENGTH}")
    f.write(f", val_split={VAL_SPLIT}, clip_grad_norm={CLIP_GRAD_NORM}\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'epoch':>6} | {'train_loss':>10} | {'val_loss':>10} | "
            f"{'lr':>12} | {'grad_norm':>10} | {'time':>20}\n")
    f.write("-" * 80 + "\n")

with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "lr",
                      "grad_norm", "timestamp"])

best_val_loss = float('inf')
patience_counter = 0

print("開始訓練模型...")
print("=" * 50)

for epoch in range(TOTAL_EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_batches = 0

    for batch_inputs, batch_targets in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        predictions, _ = model(batch_inputs)
        loss = loss_function(predictions, batch_targets)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CLIP_GRAD_NORM
        ).item()

        optimizer.step()

        train_loss_sum += loss.item()
        train_batches += 1

    avg_train_loss = train_loss_sum / train_batches

    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            predictions, _ = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            val_loss_sum += loss.item()
            val_batches += 1

    avg_val_loss = val_loss_sum / val_batches

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(avg_val_loss)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_txt_path, "a", encoding="utf-8") as f:
        f.write(f"{epoch+1:>6} | {avg_train_loss:>10.4f} | {avg_val_loss:>10.4f} | "
                f"{current_lr:>12.8f} | {grad_norm:>10.4f} | {timestamp:>20}\n")

    with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}",
                          f"{current_lr:.8f}", f"{grad_norm:.4f}", timestamp])

    if (epoch + 1) % 15 == 0:
        print(f"第 {epoch+1:4d} / {TOTAL_EPOCHS} 回合 | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.8f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n[!] Val Loss 連續 {PATIENCE} 回合未改善，提前結束！(第 {epoch+1} 回合)")
            break

print("=" * 50)
print("訓練完成！")
print(f"最佳 Val Loss: {best_val_loss:.4f}")
print(f"訓練日誌已儲存至: {log_txt_path} 與 {log_csv_path}\n")


def load_model(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"[v] 成功載入模型權重: {checkpoint_path}")
    else:
        print(f"[!] 找不到權重檔案: {checkpoint_path}")
    return model


print("\n--- 讀檔示範 ---")
new_model = MyLanguageModel(vocab_size, embed_size=256, hidden_size=256,
                             num_layers=2, dropout=0.1)
new_model = new_model.to(device)
new_model = load_model(new_model, CHECKPOINT_PATH)


def top_k_filter(logits, k=5):
    if k <= 0:
        return logits
    values, indices = torch.topk(logits, k)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(0, indices, values)
    return mask


def generate_text(model, start_word, word_to_id, id_to_word,
                  max_length=15, temperature=0.8, top_k=5):
    model.eval()

    start_id = word_to_id.get(start_word, 0)
    generated_words = [id_to_word[start_id]]

    current_input = torch.tensor([[start_id]], device=next(model.parameters()).device)
    current_hidden = None

    with torch.no_grad():
        for _ in range(max_length):
            output, current_hidden = model(current_input, current_hidden)

            logits = output.squeeze(0).squeeze(0)

            if top_k > 0:
                logits = top_k_filter(logits, k=top_k)

            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)

            next_id = torch.multinomial(probabilities, 1).item()
            next_word = id_to_word[next_id]
            generated_words.append(next_word)
            current_input = torch.tensor([[next_id]], device=current_input.device)

    return "".join(generated_words)


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
