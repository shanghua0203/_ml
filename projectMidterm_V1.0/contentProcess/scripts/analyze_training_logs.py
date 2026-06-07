#!/usr/bin/env python3
"""
訓練日誌分析腳本 — 讀取 training_log.jsonl，繪製四維度圖表並輸出統計摘要。

用法：
  python scripts/analyze_training_logs.py
  python scripts/analyze_training_logs.py --log_file data/processed/training_log.jsonl --output data/processed/training_analysis.png
"""

import argparse
import json
import math
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# ============================================================
# 全域設定：學術圖表風格 + 中文字型
# ============================================================
# 先設定 seaborn 主題，再覆蓋中文字型（順序重要：sns 會重置 rcParams）
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["axes.unicode_minus"] = False  # 修正負號顯示
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
COLORS = sns.color_palette("deep", 4)


# ============================================================
# 步驟一：命令列參數
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Mamba 訓練日誌分析與繪圖")
    parser.add_argument(
        "--log_file",
        type=str,
        default="data/processed/training_log.jsonl",
        help="訓練日誌 JSONL 檔案路徑（預設：data/processed/training_log.jsonl）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/training_analysis.png",
        help="輸出圖表路徑（預設：data/processed/training_analysis.png）",
    )
    return parser.parse_args()


# ============================================================
# 步驟二：讀取與前處理日誌
# ============================================================
def load_log(filepath):
    """
    讀取 JSONL 日誌檔，回傳清洗後的 pandas DataFrame。

    處理邏輯：
    1. 逐行讀取 JSON
    2. 依 step 去重複（保留最後一筆，解決某些 step 出現兩次的問題）
    3. 依 step 遞增排序
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # 確保必要欄位存在
    required_cols = {"step", "loss", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"日誌缺少必要欄位：{missing}")

    # 去重複：同一個 step 保留最後一筆
    df = df.drop_duplicates(subset="step", keep="last").copy()

    # 排序
    df = df.sort_values("step").reset_index(drop=True)

    # 解析 timestamp 為 datetime 型別
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 計算 PPL
    df["perplexity"] = df["loss"].apply(lambda x: math.exp(x))

    # 計算每步耗時（與前一筆的時間差，單位：秒）
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds()

    return df


# ============================================================
# 步驟三：繪製各圖表
# ============================================================

def plot_loss(ax, df):
    """圖 1：訓練損失曲線"""
    ax.plot(df["step"], df["loss"], color=COLORS[0], alpha=0.5, linewidth=0.8, label="原始 Loss")

    # 滾動平均平滑曲線（window=10）
    df["loss_smooth"] = df["loss"].rolling(window=10, min_periods=1).mean()
    ax.plot(df["step"], df["loss_smooth"], color=COLORS[0], linewidth=2, label="平滑 Loss")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("訓練損失曲線 (Training Loss)")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_ppl(ax, df):
    """圖 2：困惑度曲線（PPL = exp(loss)），Y 軸取 log scale"""
    ax.plot(df["step"], df["perplexity"], color=COLORS[1], linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity (log scale)")
    ax.set_title("困惑度曲線 (Perplexity)")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_lr(ax, df):
    """
    圖 3：學習率線性衰減模擬。

    根據 HuggingFace Trainer 預設的線性衰減：
      lr(t) = initial_lr * (1 - t / total_steps)
    已知實際訓練跑了 3840 步，初始 LR = 1e-4，最終 LR = 0。
    """
    initial_lr = 1e-4
    min_step = df["step"].min()
    max_step = df["step"].max()

    # 從 step 0 到 max_step 產生連續的步數
    steps_lr = np.linspace(0, max_step, max_step + 1)
    lr_values = initial_lr * (1 - steps_lr / max_step)
    lr_values = np.clip(lr_values, 0, initial_lr)

    ax.plot(steps_lr, lr_values, color=COLORS[2], linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"學習率變化 (Linear Decay, init={initial_lr})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_speed(ax, df):
    """圖 4：訓練速度分析 — 每 10 步的耗時"""
    # 第一筆沒有前一筆可比較，time_diff 為 NaN，跳過
    speed_data = df.dropna(subset=["time_diff"])

    ax.scatter(speed_data["step"], speed_data["time_diff"],
               color=COLORS[3], alpha=0.5, s=15, label="每 10 步耗時")

    # 平均耗時水平線
    mean_time = speed_data["time_diff"].mean()
    ax.axhline(y=mean_time, color=COLORS[3], linestyle="--", linewidth=1.5,
               label=f"平均: {mean_time:.2f} 秒")

    ax.set_xlabel("Step")
    ax.set_ylabel("時間 (秒 / 每 10 步)")
    ax.set_title("訓練速度分析 (Training Speed)")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# ============================================================
# 步驟四：印出 Markdown 統計摘要表
# ============================================================
def print_summary(df):
    """計算各項統計指標，以 Markdown 表格格式輸出"""
    final_loss = df["loss"].iloc[-1]
    final_ppl = df["perplexity"].iloc[-1]
    initial_loss = df["loss"].iloc[0]
    max_step = df["step"].max()
    total_epochs = df["epoch"].max() if "epoch" in df.columns else "未知"

    # 平均每步耗時（排除 NaN）
    speed_data = df["time_diff"].dropna()
    mean_time_per_10steps = speed_data.mean()
    mean_time_per_step = mean_time_per_10steps / 10
    estimated_total_seconds = mean_time_per_step * max_step
    estimated_total_minutes = estimated_total_seconds / 60

    # 損失下降幅度
    loss_drop = initial_loss - final_loss
    loss_drop_pct = (loss_drop / initial_loss) * 100

    summary = f"""
## 訓練統計摘要

| 指標 | 數值 |
|------|------|
| 初始 Loss | {initial_loss:.4f} |
| 最終 Loss | {final_loss:.4f} |
| Loss 下降幅度 | {loss_drop:.4f} ({loss_drop_pct:.1f}%) |
| 最終 Perplexity | {final_ppl:.4f} |
| 初始學習率 | 1e-4 |
| 總訓練步數 | {int(max_step):,} |
| 總 Epochs | {total_epochs} |
| 平均每步耗時 | {mean_time_per_step:.4f} 秒 |
| 平均每 10 步耗時 | {mean_time_per_10steps:.2f} 秒 |
| 預估總訓練時間 | {estimated_total_minutes:.1f} 分鐘 |
"""
    print(summary)


# ============================================================
# 步驟五：主程式
# ============================================================
def main():
    args = parse_args()

    # 讀取資料
    print(f"[INFO] 讀取日誌檔：{args.log_file}")
    df = load_log(args.log_file)
    print(f"[INFO] 共 {len(df)} 筆紀錄（step {df['step'].min()} ~ {df['step'].max()}）\n")

    # 建立 2×2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mamba LoRA 微調訓練分析", fontsize=16, fontweight="bold", y=1.02)

    plot_loss(axes[0, 0], df)
    plot_ppl(axes[0, 1], df)
    plot_lr(axes[1, 0], df)
    plot_speed(axes[1, 1], df)

    plt.tight_layout()

    # 儲存圖片
    print(f"[INFO] 儲存圖表至：{args.output}")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print("[INFO] 圖表儲存完成！\n")

    # 顯示圖片（若在互動環境）
    plt.show()

    # 印出統計摘要
    print_summary(df)


if __name__ == "__main__":
    main()
