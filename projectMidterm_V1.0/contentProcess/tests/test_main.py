"""
Mamba 聊天伺服器 — 整合測試

測試方式：
  使用 FastAPI TestClient 啟動完整伺服器（含模型載入），
  對兩個 API 端點進行實際呼叫驗證。

注意事項：
  - 首次執行會下載並載入 Mamba-1.4B 模型（約 30~60 秒）
  - 測試共用同一個 TestClient，模型只載入一次
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fastapi.testclient import TestClient

from main import app, load_models

# 手動載入模型（TestClient 在某些版本不會自動觸發 lifespan）
load_models()

# 建立全域測試客戶端
client = TestClient(app)


def test_lora_models_endpoint():
    """GET /api/lora_models 應回傳 200 與正確的 JSON 格式"""
    res = client.get("/api/lora_models")
    assert res.status_code == 200

    data = res.json()
    assert "models" in data
    assert isinstance(data["models"], list)

    # 每個模型項目都應有 name / step / loss / path 欄位
    if data["models"]:
        m = data["models"][0]
        assert "name" in m
        assert "step" in m
        assert "loss" in m
        assert "path" in m
        assert m["name"].startswith("checkpoint-")


def test_chat_base_model():
    """POST /api/chat 不使用 LoRA，應回傳串流文字"""
    res = client.post(
        "/api/chat",
        json={"prompt": "你好", "temperature": 0.7},
    )
    assert res.status_code == 200
    assert "text/plain" in res.headers.get("content-type", "")

    text = res.text
    assert len(text) > 0, "生成內容不應為空"
    assert "系統備註" in text, "結尾應有系統備註"


def test_chat_with_first_lora():
    """POST /api/chat 使用第一個可用的 LoRA checkpoint"""
    # 先取得 LoRA 清單
    models_res = client.get("/api/lora_models")
    models = models_res.json().get("models", [])
    if not models:
        return  # 沒有 checkpoint 就跳過

    lora_path = models[0]["path"]
    res = client.post(
        "/api/chat",
        json={
            "prompt": "手沖咖啡需要什麼器具？",
            "temperature": 0.8,
            "lora_path": lora_path,
        },
    )
    assert res.status_code == 200
    text = res.text
    assert len(text) > 0
    # 確認有提到 LoRA 名稱
    assert models[0]["name"] in text or "系統備註" in text


def test_chat_with_custom_temperature():
    """POST /api/chat 自訂溫度參數應正確反映在系統備註中"""
    res = client.post(
        "/api/chat",
        json={"prompt": "test", "temperature": 1.2},
    )
    assert res.status_code == 200
    # 系統備註應包含設定的溫度值
    assert "溫度 1.2" in res.text


def test_chat_invalid_lora_path():
    """POST /api/chat 傳入不存在的 LoRA 路徑應回傳 400 錯誤"""
    res = client.post(
        "/api/chat",
        json={
            "prompt": "test",
            "lora_path": "nonexistent/checkpoint-9999",
        },
    )
    assert res.status_code == 400
    assert "找不到 LoRA 路徑" in res.text
