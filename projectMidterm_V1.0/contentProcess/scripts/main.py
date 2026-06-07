#!/usr/bin/env python3
"""
Mamba 模型聊天網頁 — FastAPI 後端伺服器

提供兩個 API：
  1. GET  /api/lora_models   — 列出所有可用的 LoRA checkpoint
  2. POST /api/chat          — 串流文字生成（SSE 打字機效果）

啟動方式：
  source .venv/bin/activate
  python scripts/main.py
  或
  uvicorn scripts.main:app --host 0.0.0.0 --port 8080 --reload
"""

import json
import os
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ============================================================
# 路徑設定
# ============================================================
# 專案根目錄（這個檔案在 scripts/ 底下，所以要往上一層）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_BASE_DIR = os.path.join(BASE_DIR, "coffee_mamba_lora")

# ============================================================
# 全域變數 — 伺服器啟動時載入，常駐記憶體
# ============================================================
model = None      # 基礎 Mamba 模型，之後會依需求掛載 LoRA
base_model = None # 純基礎模型參照（永不掛 LoRA，確保無權重汙染）
tokenizer = None  # 分詞器
device = None     # 自動偵測到的裝置（cuda / mps / cpu）

# ============================================================
# 建立 FastAPI 實體
# ============================================================
# ============================================================
# 應用程式生命週期（啟動與關閉）
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """伺服器啟動時自動載入模型，關閉時可釋放資源"""
    load_models()
    yield
    # 關閉時可加入清理邏輯（例如釋放 GPU 記憶體）
    print("[INFO] 伺服器關閉")


app = FastAPI(title="Mamba LoRA 聊天伺服器", lifespan=lifespan)


# ============================================================
# API 請求的資料格式（Pydantic 自動驗證）
# ============================================================
class ChatRequest(BaseModel):
    """
    前端傳來的聊天請求格式：
      prompt:      使用者的提問文字
      temperature: 生成溫度（0.1 ~ 1.5）
      lora_path:   LoRA 權重資料夾路徑，不傳或 null 代表只用基礎模型
    """
    prompt: str
    temperature: float = 0.7
    lora_path: str | None = None


# ============================================================
# 硬體自動偵測（跟 mamba_inference.py 同樣邏輯）
# ============================================================
def auto_device():
    if torch.cuda.is_available():
        dev = "cuda"
        print(f"[INFO] 使用 CUDA（{torch.cuda.get_device_name(0)}）")
    elif torch.backends.mps.is_available():
        dev = "mps"
        print("[INFO] 使用 Apple MPS")
    else:
        dev = "cpu"
        print("[INFO] 使用 CPU")
    return dev


# ============================================================
# 載入模型（在 lifespan 啟動時呼叫）
# ============================================================
def load_models():
    """
    伺服器啟動時自動執行：
    1. 偵測硬體裝置
    2. 載入 Mamba 基礎模型到記憶體（常駐，不重複載入）
    3. 載入分詞器
    """
    global model, base_model, tokenizer, device

    device = auto_device()

    MODEL_NAME = "state-spaces/mamba-1.4b-hf"

    # 載入分詞器
    print(f"[INFO] 正在載入分詞器：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 載入基礎模型
    print(f"[INFO] 正在載入基礎模型：{MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    base_model = model
    print(f"[INFO] 模型成功載入，參數量：{sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# API 1：取得所有 LoRA checkpoint 清單
# ============================================================
@app.get("/api/lora_models")
def list_lora_models():
    """
    掃描 coffee_mamba_lora/ 資料夾，找出所有 checkpoint-* 子資料夾。
    每個 checkpoint 讀取 trainer_state.json，提取 global_step 與最新 loss。

    回傳格式：
    {
      "models": [
        {
          "name": "checkpoint-2112",
          "step": 2112,
          "loss": 8.14,
          "path": "coffee_mamba_lora/checkpoint-2112"
        },
        ...
      ]
    }
    """
    models_list = []

    # 檢查 LoRA 資料夾是否存在
    if not os.path.isdir(LORA_BASE_DIR):
        return {"models": []}

    # 掃描所有 checkpoint- 開頭的資料夾
    for entry in os.scandir(LORA_BASE_DIR):
        if not entry.is_dir() or not entry.name.startswith("checkpoint-"):
            continue

        checkpoint_path = entry.path
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")

        step = None
        loss = None

        # 嘗試讀取 trainer_state.json
        if os.path.isfile(trainer_state_path):
            try:
                with open(trainer_state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                step = state.get("global_step")
                # 從 log_history 取得最後一筆 loss
                log_history = state.get("log_history", [])
                if log_history:
                    loss = log_history[-1].get("loss")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        models_list.append({
            "name": entry.name,
            "step": step if step is not None else "未知",
            "loss": round(loss, 4) if loss is not None else "未知",
            "path": os.path.join("coffee_mamba_lora", entry.name),
        })

    # 依照 step 數字遞減排序（最新的在前面）
    def sort_key(m):
        s = m["step"]
        return s if isinstance(s, (int, float)) else 0
    models_list.sort(key=sort_key, reverse=True)

    return {"models": models_list}


# ============================================================
# API 2：串流聊天生成（Server-Sent Events）
# ============================================================
@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    """
    接收使用者的提問，用 Mamba 模型生成回答，並用串流方式逐字回傳。

    處理流程：
    1. 若有指定 lora_path，用 PeftModel.from_pretrained 動態載入 LoRA 權重
    2. 將 prompt 轉為 tensor
    3. 用 TextIteratorStreamer 搭配 Thread 非同步產生文字
    4. 逐 token yield 給前端（打字機效果）
    5. 最後 yield 系統備註
    """
    global model, base_model, tokenizer, device

    # --- 步驟 1：選擇性載入 LoRA 權重 ---
    if req.lora_path:
        lora_full_path = os.path.join(BASE_DIR, req.lora_path)
        if not os.path.isdir(lora_full_path):
            raise HTTPException(status_code=400, detail=f"找不到 LoRA 路徑：{req.lora_path}")

        try:
            if not isinstance(model, PeftModel):
                # 第一次使用 LoRA：從純基礎模型建立 PeftModel
                model = PeftModel.from_pretrained(base_model, lora_full_path)
                model.to(device)
                model.eval()
            else:
                # 後續切換：覆蓋同一 adapter slot 的權重 + 明確切換
                model.load_adapter(lora_full_path, adapter_name="default")
                model.set_adapter("default")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"載入 LoRA 失敗：{str(e)}")

        active_model = model
        lora_folder_name = os.path.basename(lora_full_path)
    else:
        active_model = base_model
        lora_folder_name = "基礎模型（無 LoRA）"

    # --- 步驟 2：準備輸入 ---
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    # --- 步驟 3：設定 TextIteratorStreamer（打字機效果） ---
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,       # 只輸出生成的內容，不重複 prompt
        skip_special_tokens=True,
    )

    # 生成參數
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=req.temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # --- 步驟 4：在背景執行緒執行生成 ---
    # TextIteratorStreamer 需要搭配 thread，因為 model.generate 是同步的
    thread = threading.Thread(target=active_model.generate, kwargs=generation_kwargs)
    thread.start()

    # --- 步驟 5：定義串流生成器（SSE 格式） ---
    async def text_generator():
        # 逐 token 從 streamer 取出並 yield
        for text in streamer:
            yield text

        # 所有 token 生成完後，附加系統備註
        note = (
            f"\n\n> 💡 系統備註：本回答產生於 溫度 {req.temperature}，"
            f"使用權重 {lora_folder_name}"
        )
        yield note

        # 等待生成執行緒結束
        thread.join()

    # 回傳 SSE 串流回應
    return StreamingResponse(
        text_generator(),
        media_type="text/plain; charset=utf-8",
    )


# ============================================================
# 掛載靜態檔案（前端 index.html）
# ============================================================
# 將 static/ 資料夾對應到 /static 路徑
static_dir = os.path.join(BASE_DIR, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


# ============================================================
# 啟動點
# ============================================================
if __name__ == "__main__":
    import sys
    # 將專案根目錄加入 sys.path，使 uvicorn 能找到 scripts.main
    sys.path.insert(0, BASE_DIR)
    import uvicorn
    uvicorn.run("scripts.main:app", host="0.0.0.0", port=8080, reload=False)
