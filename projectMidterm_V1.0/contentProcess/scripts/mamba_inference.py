#!/usr/bin/env python3
"""
Mamba 模型推理腳本 — 支援 LoRA 權重載入與互動聊天模式

用法範例：
  # 只用基礎模型
  python scripts/mamba_inference.py

  # 載入 LoRA 權重
  python scripts/mamba_inference.py --use_lora --lora_path ./coffee_mamba_lora/checkpoint-3840

  # 自訂生成參數
  python scripts/mamba_inference.py --temperature 0.8 --max_tokens 300 --top_p 0.95
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 步驟一：命令列參數解析（argparse）
# ============================================================
def parse_args(args=None):
    """
    解析使用者在命令列輸入的參數。
    所有參數都有預設值，不傳任何參數也能直接執行。

    參數 args：可供測試時傳入字串串列來模擬命令列輸入，不傳則讀取 sys.argv。
    """
    parser = argparse.ArgumentParser(
        description="Mamba 模型推理腳本 — 支援 LoRA 載入與互動聊天模式"
    )

    # --- 模型與 LoRA 設定 ---
    parser.add_argument(
        "--model_name",
        type=str,
        default="state-spaces/mamba-1.4b-hf",
        help="Hugging Face 上的基礎模型名稱（預設：state-spaces/mamba-1.4b-hf）",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="啟用這個參數就會載入 LoRA 權重（不加這個參數就不載入）",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./coffee_mamba_lora",
        help="LoRA 權重的資料夾路徑（預設：./coffee_mamba_lora）",
    )

    # --- 生成參數 ---
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="溫度參數，越高隨機性越強（預設：0.7）",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="每次最多生成多少個新 token（預設：200）",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p 採樣門檻，用來過濾掉太奇怪的詞彙（預設：0.9）",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="重複懲罰，>1.0 可以避免模型一直講重複的話（預設：1.1）",
    )

    return parser.parse_args(args=args)


# ============================================================
# 步驟二：硬體自動偵測
# ============================================================
def auto_device():
    """
    自動判斷要用哪個裝置跑模型：
    1. 有 NVIDIA GPU → CUDA（最快）
    2. 有 Apple 晶片 → MPS（次快）
    3. 都沒有 → CPU（最慢但一定可以跑）
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[INFO] 使用 CUDA（{torch.cuda.get_device_name(0)}）")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[INFO] 使用 Apple MPS")
    else:
        device = "cpu"
        print("[INFO] 使用 CPU")
    return device


# ============================================================
# 步驟三：載入模型與分詞器
# ============================================================
def load_model_and_tokenizer(model_name, use_lora, lora_path, device):
    """
    載入基礎模型 + 分詞器，選擇性掛載 LoRA 權重。

    流程：
    1. 載入分詞器（Tokenizer）— 把文字轉成 token
    2. 載入基礎模型（Base Model）— Mamba 本體
    3. 如果有指定 --use_lora，就用 PeftModel 掛上額外訓練的 LoRA 權重
    4. 把模型搬到對應的裝置（GPU / MPS / CPU）
    5. 切換成評估模式（eval），關掉 dropout 等訓練行為
    """
    # --- 載入分詞器 ---
    # 分詞器負責把人類的文字切成一塊塊的數字（token），餵給模型吃
    print(f"[INFO] 正在載入分詞器：{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Mamba 原廠分詞器沒有設定 pad_token，手動設成 eos_token 避免報錯
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 載入基礎模型 ---
    # AutoModelForCausalLM 會自動認出這是 Mamba 架構，不用手動指定
    print(f"[INFO] 正在載入基礎模型：{model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # --- 選擇性載入 LoRA 權重 ---
    if use_lora:
        print(f"[INFO] 偵測到 --use_lora，正在載入 LoRA 權重：{lora_path}")
        from peft import PeftModel

        # PeftModel.from_pretrained 會把訓練好的 LoRA 外掛權重疊到基礎模型上
        model = PeftModel.from_pretrained(model, lora_path)
        print("[INFO] LoRA 權重載入成功！")

    # --- 搬到目標裝置 + 切換成評估模式 ---
    model = model.to(device)
    model.eval()

    # 印出模型總參數量，方便確認有沒有正確載入
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型成功載入，參數量：{total_params:,}")

    return model, tokenizer


# ============================================================
# 步驟四：單次生成回應
# ============================================================
def generate_response(model, tokenizer, prompt, args, device):
    """
    給一段文字 prompt，讓模型生成回應。
    所有的生成參數（temperature、max_tokens 等）都來自 argparse。

    參數說明：
      - do_sample=True：啟用隨機採樣，不然模型每次都會選機率最高的詞
      - temperature：控制隨機程度，越低越保守，越高越有創意
      - top_p：累積機率門檻，只保留機率加起來夠高的詞彙
      - repetition_penalty：處罰重複出現的詞，>1 代表越重複分數扣越多
      - pad_token_id：指定 padding 用的 token，避免長度不一致時報錯
    """
    # 把文字轉成 tensor，並搬到同一顆裝置上
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 把 token 解碼回人類看得懂的文字
    # 跳過特殊 token（例如 <pad>、<eos> 等）
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # 把輸入的 prompt 部分去掉，只保留模型新產生的內容
    response = response[len(prompt):].strip()

    return response


# ============================================================
# 步驟五：互動聊天模式
# ============================================================
def interactive_loop(model, tokenizer, args, device):
    """
    進入無限迴圈的聊天模式：
    - 顯示「User: 」等待使用者輸入
    - 輸入 exit 或 quit 就會結束程式
    - 其他內容都當作問題送給模型生成回答
    """
    print("\n" + "=" * 50)
    print("  進入互動聊天模式！輸入 exit 或 quit 結束")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            # 遇到 Ctrl+D 或 Ctrl+C 也優雅地結束
            print("\n[INFO] 偵測到中斷訊號，結束程式")
            break

        if user_input.lower() in ("exit", "quit"):
            print("[INFO] 結束互動模式，掰掰～")
            break

        if not user_input:
            # 跳過空白輸入
            continue

        # 讓模型生成回應
        response = generate_response(model, tokenizer, user_input, args, device)
        print(f"Assistant: {response}")


# ============================================================
# 步驟六：主程式入口
# ============================================================
def main():
    """
    主要流程：
    1. 解析命令列參數
    2. 自動偵測硬體
    3. 載入模型（含選擇性 LoRA）
    4. 進入互動聊天模式
    """
    args = parse_args()
    device = auto_device()
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_path=args.lora_path,
        device=device,
    )
    interactive_loop(model, tokenizer, args, device)


# 只有直接執行這個檔案時才會跑 main()
# 如果是被 import 就不會自動執行
if __name__ == "__main__":
    main()
