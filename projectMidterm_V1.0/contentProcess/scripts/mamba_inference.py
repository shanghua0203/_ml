import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 設定區：可以在這邊自由切換模型
# ============================================================
# Hugging Face 上官方提供的 Mamba 模型（已轉換為 transformers 相容格式）
# 可選：mamba-130m-hf, mamba-370m-hf, mamba-790m-hf, mamba-1.4b-hf, mamba-2.8b-hf
MODEL_NAME = "state-spaces/mamba-1.4b-hf"

# ============================================================
# 步驟一：硬體自動偵測
# ============================================================
# 檢查有沒有 NVIDIA GPU，有的話用 CUDA
if torch.cuda.is_available():
    device = "cuda"
    print(f"[INFO] 使用 CUDA（{torch.cuda.get_device_name(0)}）")
# 檢查有沒有 Apple 晶片，有的話用 MPS
elif torch.backends.mps.is_available():
    device = "mps"
    print("[INFO] 使用 Apple MPS")
# 都沒有就乖乖用 CPU
else:
    device = "cpu"
    print("[INFO] 使用 CPU")

# ============================================================
# 步驟二：載入分詞器（Tokenizer）
# ============================================================
# 分詞器負責把文字切成一塊塊的 Token，餵給模型吃
print(f"[INFO] 正在載入分詞器：{MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# 步驟三：載入模型本體
# ============================================================
# AutoModelForCausalLM 會自動認出這是 Mamba 架構，不用手動指定
print(f"[INFO] 正在載入模型：{MODEL_NAME}（首次會下載約 500MB）")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print(f"[INFO] 模型成功載入，參數量：{sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 步驟四：Wake-up Test — 確認模型能正常思考
# ============================================================
print("\n" + "=" * 50)
print("  Wake-up Test：讓 Mamba 模型說說話")
print("=" * 50)

input_text = "你好"
print(f"輸入：{input_text}")

# 把文字轉成 Tensor，並搬到同一顆裝置上
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 讓模型生成回應
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

# 把 Token 解碼回人類看得懂的文字
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"輸出：{response}")
print("=" * 50)
