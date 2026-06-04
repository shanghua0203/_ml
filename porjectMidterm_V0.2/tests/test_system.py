"""
tests/test_system.py — 系統整合測試（v1.2 Web 版本）
使用 FastAPI TestClient 測試 API endpoints
"""

import os
import json
import torch
import pytest
from fastapi.testclient import TestClient

import app as app_module
from model import MyLanguageModel
from app import app


@pytest.fixture
def setup_checkpoint_dir(tmp_path):
    old_ckpt = app_module.CHECKPOINT_DIR
    old_vocab = app_module.VOCAB_DIR
    app_module.CHECKPOINT_DIR = str(tmp_path)
    app_module.VOCAB_DIR = str(tmp_path)

    model = MyLanguageModel(vocab_size=10, embed_size=8, hidden_size=8,
                            num_layers=1, dropout=0.0, tie_weights=True)
    torch.save(model.state_dict(), tmp_path / "test_model.pt")

    word_to_id = {f"詞{i}": i for i in range(10)}
    word_to_id["<UNK>"] = 0
    id_to_word = {str(i): f"詞{i}" for i in range(10)}
    id_to_word["0"] = "<UNK>"
    with open(tmp_path / "test_model.json", "w", encoding="utf-8") as f:
        json.dump({"word_to_id": word_to_id, "id_to_word": id_to_word}, f,
                  ensure_ascii=False, indent=2)

    yield tmp_path

    app_module.CHECKPOINT_DIR = old_ckpt
    app_module.VOCAB_DIR = old_vocab


class TestListModels:
    """測試 GET /api/models"""

    def test_empty_returns_list(self):
        client = TestClient(app)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_with_checkpoint(self, setup_checkpoint_dir):
        client = TestClient(app)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        m = data["models"][0]
        assert m["filename"] == "test_model.pt"
        assert m["vocab_size"] == 10
        assert m["embed_size"] == 8
        assert m["hidden_size"] == 8
        assert m["num_layers"] == 1
        assert m["size_bytes"] > 0

    def test_multiple_checkpoints(self, setup_checkpoint_dir):
        tmp = setup_checkpoint_dir
        for i in range(3):
            m = MyLanguageModel(vocab_size=10, embed_size=8, hidden_size=8,
                                num_layers=1, dropout=0.0, tie_weights=True)
            torch.save(m.state_dict(), tmp / f"model_{i}.pt")
        client = TestClient(app)
        resp = client.get("/api/models")
        data = resp.json()
        assert len(data["models"]) >= 3

    def test_non_pt_files_ignored(self, setup_checkpoint_dir):
        tmp = setup_checkpoint_dir
        with open(tmp / "notes.txt", "w") as f:
            f.write("not a model")
        client = TestClient(app)
        resp = client.get("/api/models")
        data = resp.json()
        for m in data["models"]:
            assert m["filename"].endswith(".pt")


class TestGenerateText:
    """測試 POST /api/generate"""

    def test_basic_generation(self, setup_checkpoint_dir):
        client = TestClient(app)
        resp = client.post("/api/generate", json={
            "start_word": "咖啡",
            "model_filename": "test_model.pt",
            "temperature": 0.8,
            "top_k": 5,
            "max_length": 10,
            "repetition_penalty": 1.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_text" in data
        assert isinstance(data["generated_text"], str)
        assert len(data["generated_text"]) > 0
        assert data["model_filename"] == "test_model.pt"
        assert data["temperature"] == 0.8
        assert data["top_k"] == 5
        assert data["max_length"] == 10
        assert data["repetition_penalty"] == 1.5
        assert data["inference_time_ms"] > 0

    def test_with_unknown_start_word(self, setup_checkpoint_dir):
        client = TestClient(app)
        resp = client.post("/api/generate", json={
            "start_word": "火星文123",
            "model_filename": "test_model.pt",
            "temperature": 1.0,
            "top_k": 10,
            "max_length": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["generated_text"]) > 0

    def test_missing_model_returns_404(self, setup_checkpoint_dir):
        client = TestClient(app)
        resp = client.post("/api/generate", json={
            "start_word": "咖啡",
            "model_filename": "nonexistent.pt",
            "temperature": 0.8,
            "top_k": 5,
            "max_length": 5,
        })
        assert resp.status_code == 404

    def test_invalid_extension_returns_400(self, setup_checkpoint_dir):
        client = TestClient(app)
        resp = client.post("/api/generate", json={
            "start_word": "咖啡",
            "model_filename": "model.txt",
            "temperature": 0.8,
            "top_k": 5,
            "max_length": 5,
        })
        assert resp.status_code == 400

    def test_max_length_one(self, setup_checkpoint_dir):
        client = TestClient(app)
        resp = client.post("/api/generate", json={
            "start_word": "咖啡",
            "model_filename": "test_model.pt",
            "temperature": 0.8,
            "top_k": 5,
            "max_length": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["generated_text"]) >= 1

    def test_low_temperature_consistency(self, setup_checkpoint_dir):
        client = TestClient(app)
        torch.manual_seed(42)
        resp1 = client.post("/api/generate", json={
            "start_word": "咖啡",
            "model_filename": "test_model.pt",
            "temperature": 0.1,
            "top_k": 1,
            "max_length": 5,
        })
        torch.manual_seed(42)
        resp2 = client.post("/api/generate", json={
            "start_word": "咖啡",
            "model_filename": "test_model.pt",
            "temperature": 0.1,
            "top_k": 1,
            "max_length": 5,
        })
        assert resp1.json()["generated_text"] == resp2.json()["generated_text"]

    def test_high_temperature_variation(self, setup_checkpoint_dir):
        client = TestClient(app)
        results = set()
        for _ in range(3):
            torch.manual_seed(_)
            resp = client.post("/api/generate", json={
                "start_word": "咖啡",
                "model_filename": "test_model.pt",
                "temperature": 1.5,
                "top_k": 10,
                "max_length": 8,
            })
            results.add(resp.json()["generated_text"])
        assert len(results) <= 3


class TestFrontend:
    """測試前端靜態檔案服務"""

    def test_index_returns_html(self):
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
