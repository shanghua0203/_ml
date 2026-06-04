"""
tests/test_unit.py — 單元測試（v1.2 Web 版本）
測試 app.py 中的輔助函數與推論邏輯
"""

import os
import json
import tempfile
import torch
import torch.nn as nn
import pytest

from model import MyLanguageModel
from app import infer_model_arch, load_vocab, load_model_from_checkpoint


class TestInferModelArch:
    """測試從 state_dict 推斷模型架構的邏輯"""

    def test_infer_from_real_checkpoint(self):
        real_pt = os.path.join(os.path.dirname(__file__), "..",
                               "model_checkpoint.pt")
        if not os.path.exists(real_pt):
            pytest.skip("找不到 model_checkpoint.pt，跳過真實檢查點測試")
        state = torch.load(real_pt, map_location="cpu", weights_only=True)
        vs, es, hs, nl = infer_model_arch(state)
        assert vs > 0
        assert es == hs
        assert nl >= 1
        assert vs == state["embedding.weight"].shape[0]
        assert es == state["embedding.weight"].shape[1]

    def test_infer_from_minimal_state_dict(self):
        state = {
            "embedding.weight": torch.randn(10, 8),
            "lstm.weight_ih_l0": torch.randn(32, 8),
            "lstm.weight_hh_l0": torch.randn(32, 8),
            "lstm.bias_ih_l0": torch.randn(32),
            "lstm.bias_hh_l0": torch.randn(32),
        }
        vs, es, hs, nl = infer_model_arch(state)
        assert vs == 10
        assert es == 8
        assert hs == 8
        assert nl == 1

    def test_infer_multi_layer(self):
        state = {
            "embedding.weight": torch.randn(100, 32),
            "lstm.weight_ih_l0": torch.randn(128, 32),
            "lstm.weight_hh_l0": torch.randn(128, 32),
            "lstm.bias_ih_l0": torch.randn(128),
            "lstm.bias_hh_l0": torch.randn(128),
            "lstm.weight_ih_l1": torch.randn(128, 32),
            "lstm.weight_hh_l1": torch.randn(128, 32),
            "lstm.bias_ih_l1": torch.randn(128),
            "lstm.bias_hh_l1": torch.randn(128),
            "fc.weight": torch.randn(100, 32),
        }
        vs, es, hs, nl = infer_model_arch(state)
        assert vs == 100
        assert es == 32
        assert hs == 32
        assert nl == 2


class TestLoadVocab:
    """測試從 JSON 載入詞彙表"""

    def test_load_vocab_roundtrip(self):
        word_to_id = {"<UNK>": 0, "咖啡": 1, "沖泡": 2, "溫度": 3}
        id_to_word = {"0": "<UNK>", "1": "咖啡", "2": "沖泡", "3": "溫度"}
        data = {"word_to_id": word_to_id, "id_to_word": id_to_word}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         encoding="utf-8", delete=False) as f:
            json.dump(data, f, ensure_ascii=False)
            tmp = f.name

        try:
            w2i, i2w = load_vocab(tmp)
            assert w2i["咖啡"] == 1
            assert w2i["<UNK>"] == 0
            assert i2w[3] == "溫度"
            assert len(w2i) == 4
            assert len(i2w) == 4
        finally:
            os.unlink(tmp)


class TestLoadModelFromCheckpoint:
    """測試從檢查點載入模型"""

    def test_save_and_load_minimal_model(self, tmp_path):
        ckpt = tmp_path / "test_model.pt"
        model = MyLanguageModel(vocab_size=10, embed_size=8, hidden_size=8,
                                num_layers=1, dropout=0.0, tie_weights=True)
        torch.save(model.state_dict(), ckpt)
        loaded = load_model_from_checkpoint(str(ckpt))
        assert loaded.embedding.num_embeddings == 10
        assert loaded.embedding.embedding_dim == 8
        assert loaded.lstm.hidden_size == 8
        assert loaded.lstm.num_layers == 1
        x = torch.randint(0, 10, (1, 4))
        with torch.no_grad():
            out, _ = loaded(x)
        assert out.shape == (1, 10)

    def test_save_and_load_multi_layer(self, tmp_path):
        ckpt = tmp_path / "test_model2.pt"
        model = MyLanguageModel(vocab_size=20, embed_size=16, hidden_size=16,
                                num_layers=2, dropout=0.1, tie_weights=True)
        torch.save(model.state_dict(), ckpt)
        loaded = load_model_from_checkpoint(str(ckpt))
        assert loaded.lstm.num_layers == 2
        x = torch.randint(0, 20, (2, 8))
        with torch.no_grad():
            out, _ = loaded(x)
        assert out.shape == (2, 20)

    def test_loaded_model_can_generate(self, tmp_path):
        ckpt = tmp_path / "gen_test.pt"
        model = MyLanguageModel(vocab_size=10, embed_size=8, hidden_size=8,
                                num_layers=1, dropout=0.0, tie_weights=True)
        torch.save(model.state_dict(), ckpt)
        loaded = load_model_from_checkpoint(str(ckpt))
        loaded.eval()
        with torch.no_grad():
            x = torch.tensor([[1]])
            out, _ = loaded(x)
        assert torch.isfinite(out).all()
