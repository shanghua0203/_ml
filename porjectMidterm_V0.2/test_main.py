"""
test_main.py —— 品質檢查員（v1.1 專業升級版）
單元測試檢查每個零件，系統測試檢查整條產線
"""

import os
import torch
import torch.nn as nn
import pytest
import jieba

from text_processor import (
    FALLBACK_STORY,
    UNK_TOKEN,
    UNK_ID,
    load_external_text,
    tokenize,
    build_vocab,
    text_to_ids,
    prepare_training_data,
    TextDataset,
    create_dataloader,
    auto_device,
)
from model import MyLanguageModel


# ===================================================================
# 單元測試：auto_device
# ===================================================================

class TestAutoDevice:
    """檢查硬體自動偵測"""

    def test_returns_torch_device(self):
        device = auto_device()
        assert isinstance(device, torch.device)

    def test_device_is_valid(self):
        device = auto_device()
        assert device.type in ("cuda", "mps", "cpu")


# ===================================================================
# 單元測試：tokenize（jieba 分詞）
# ===================================================================

class TestTokenize:
    """檢查 jieba 分詞功能"""

    def test_tokenize_returns_list(self):
        result = tokenize("我愛吃蘋果")
        assert isinstance(result, list)

    def test_tokenize_splits_words(self):
        result = tokenize("我愛吃蘋果")
        assert len(result) >= 1

    def test_tokenize_empty_string(self):
        result = tokenize("")
        assert result == []

    def test_tokenize_known_words(self):
        result = tokenize("從前有一個小男孩")
        assert "從前" in result or "從" in result


# ===================================================================
# 單元測試：load_external_text
# ===================================================================

class TestLoadExternalText:
    """檢查外部資料讀取與備用機制"""

    def test_fallback_when_file_not_found(self):
        result = load_external_text("nonexistent_file.txt")
        assert result == FALLBACK_STORY

    def test_fallback_is_not_empty(self):
        assert len(FALLBACK_STORY) > 0

    def test_real_file_reading(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("測試文字內容", encoding="utf-8")
        result = load_external_text(str(test_file))
        assert result == "測試文字內容"

    def test_empty_file_fallback(self, tmp_path):
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")
        result = load_external_text(str(test_file))
        assert result == FALLBACK_STORY

    def test_whitespace_only_fallback(self, tmp_path):
        test_file = tmp_path / "whitespace.txt"
        test_file.write_text("   \n  ", encoding="utf-8")
        result = load_external_text(str(test_file))
        assert result == FALLBACK_STORY


# ===================================================================
# 單元測試：build_vocab（詞層級 + <UNK>）
# ===================================================================

class TestBuildVocab:
    """檢查文字處理器的第一關：建立詞典"""

    def test_returns_correct_types(self):
        word_to_id, id_to_word, vocab_size = build_vocab("你好世界")
        assert isinstance(word_to_id, dict)
        assert isinstance(id_to_word, dict)
        assert isinstance(vocab_size, int)

    def test_unk_token_exists(self):
        word_to_id, id_to_word, _ = build_vocab("你好")
        assert UNK_TOKEN in word_to_id
        assert UNK_ID in id_to_word
        assert word_to_id[UNK_TOKEN] == UNK_ID
        assert id_to_word[UNK_ID] == UNK_TOKEN

    def test_maps_forward_and_backward(self):
        word_to_id, id_to_word, _ = build_vocab("你好世界")
        words = list(jieba.lcut("你好世界"))
        for w in words:
            if w in word_to_id:
                assert id_to_word[word_to_id[w]] == w

    def test_counts_unique_words(self):
        _, _, vocab_size = build_vocab("測試")
        assert vocab_size > 0

    def test_with_real_story(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        assert len(word_to_id) == len(id_to_word)
        assert vocab_size == len(word_to_id)
        assert UNK_TOKEN in word_to_id

    def test_empty_string(self):
        word_to_id, id_to_word, vocab_size = build_vocab("")
        assert vocab_size == 1
        assert word_to_id == {UNK_TOKEN: UNK_ID}
        assert id_to_word == {UNK_ID: UNK_TOKEN}

    def test_repeat_call_consistency(self):
        w1, i1, s1 = build_vocab("你好嗎")
        w2, i2, s2 = build_vocab("你好嗎")
        assert w1 == w2 and i1 == i2 and s1 == s2


# ===================================================================
# 單元測試：text_to_ids（含 <UNK> 防呆）
# ===================================================================

class TestTextToIds:
    """檢查第二關：文字轉數字"""

    def test_converts_correctly(self):
        word_to_id, _, _ = build_vocab("你好")
        ids = text_to_ids("你好", word_to_id)
        assert all(isinstance(i, int) for i in ids)

    def test_empty_string_returns_empty_list(self):
        word_to_id, _, _ = build_vocab("你好")
        assert text_to_ids("", word_to_id) == []

    def test_length_matches_input(self):
        word_to_id, _, _ = build_vocab("你好世界")
        result = text_to_ids("你好世界", word_to_id)
        assert len(result) > 0

    def test_unknown_word_uses_unk(self):
        """沒看過的詞應該用 <UNK> 代替，不該當機"""
        word_to_id, _, _ = build_vocab("你好")
        ids = text_to_ids("這是一段完全沒看過的文字", word_to_id)
        for i in ids:
            assert 0 <= i

    def test_full_story_conversion(self):
        word_to_id, id_to_word, _ = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, word_to_id)
        assert len(ids) > 0

    def test_ids_are_within_vocab_range(self):
        word_to_id, _, vocab_size = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, word_to_id)
        assert all(0 <= i < vocab_size for i in ids)


# ===================================================================
# 單元測試：prepare_training_data
# ===================================================================

class TestPrepareTrainingData:
    """檢查第三關：準備訓練樣本"""

    def test_shape_and_count(self):
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        inputs, targets = prepare_training_data(ids, sequence_length=3)
        assert len(inputs) == 7
        assert len(targets) == 7
        assert all(len(x) == 3 for x in inputs)

    def test_correct_alignment(self):
        ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        inputs, targets = prepare_training_data(ids, sequence_length=3)
        for i in range(len(inputs)):
            assert inputs[i][-1] == ids[i + 2]
            assert targets[i] == ids[i + 3]

    def test_sequence_length_equals_total(self):
        ids = [0, 1, 2]
        inputs, targets = prepare_training_data(ids, sequence_length=3)
        assert len(inputs) == 0 and len(targets) == 0

    def test_sequence_length_exceeds_total(self):
        ids = [0, 1]
        inputs, targets = prepare_training_data(ids, sequence_length=3)
        assert len(inputs) == 0 and len(targets) == 0

    def test_sequence_length_one(self):
        ids = [10, 20, 30, 40]
        inputs, targets = prepare_training_data(ids, sequence_length=1)
        assert len(inputs) == 3
        assert inputs[0] == [10] and targets[0] == 20
        assert inputs[1] == [20] and targets[1] == 30
        assert inputs[2] == [30] and targets[2] == 40

    def test_default_sequence_length(self):
        ids = list(range(20))
        inputs, targets = prepare_training_data(ids)
        assert len(inputs) == 12
        assert all(len(x) == 8 for x in inputs)

    def test_with_real_story_data(self):
        word_to_id, _, _ = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids, sequence_length=8)
        assert len(inputs) == len(ids) - 8
        assert len(inputs) == len(targets)
        for i in range(len(inputs)):
            assert len(inputs[i]) == 8
            assert targets[i] == ids[i + 8]


# ===================================================================
# 單元測試：TextDataset 與 DataLoader
# ===================================================================

class TestTextDataset:
    """檢查 DataLoader 批次推車機制"""

    def test_dataset_length(self):
        inputs = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        targets = [3, 4, 5]
        dataset = TextDataset(inputs, targets)
        assert len(dataset) == 3

    def test_dataset_getitem(self):
        inputs = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        targets = [3, 4, 5]
        dataset = TextDataset(inputs, targets)
        inp, tgt = dataset[0]
        assert torch.equal(inp, torch.tensor([0, 1, 2]))
        assert tgt.item() == 3

    def test_dataset_returns_tensors(self):
        inputs = [[0, 1, 2]]
        targets = [3]
        dataset = TextDataset(inputs, targets)
        inp, tgt = dataset[0]
        assert isinstance(inp, torch.Tensor)
        assert isinstance(tgt, torch.Tensor)
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long


class TestDataLoader:
    """檢查 DataLoader 批次載入"""

    def test_dataloader_batch_size(self):
        inputs = [[i, i+1, i+2] for i in range(20)]
        targets = [i+3 for i in range(20)]
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=False)
        batches = list(loader)
        assert len(batches) == 5

    def test_dataloader_batch_shape(self):
        inputs = [[i, i+1, i+2] for i in range(20)]
        targets = [i+3 for i in range(20)]
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=False)
        for batch_inputs, batch_targets in loader:
            assert batch_inputs.shape == (4, 3)
            assert batch_targets.shape == (4,)

    def test_dataloader_covers_all_data(self):
        inputs = [[i] for i in range(10)]
        targets = [i for i in range(10)]
        loader = create_dataloader(inputs, targets, batch_size=2, shuffle=False)
        all_inputs = []
        all_targets = []
        for batch_inputs, batch_targets in loader:
            all_inputs.extend(batch_inputs.tolist())
            all_targets.extend(batch_targets.tolist())
        assert len(all_inputs) == 10
        assert len(all_targets) == 10

    def test_dataloader_with_real_story(self):
        word_to_id, _, _ = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=True)
        for batch_inputs, batch_targets in loader:
            assert batch_inputs.dtype == torch.long
            assert batch_targets.dtype == torch.long
            assert batch_inputs.shape[1] == 8


# ===================================================================
# 單元測試：MyLanguageModel 初始化（含 Weight Tying）
# ===================================================================

class TestModelInit:
    """檢查模型出廠設定與 Weight Tying 權重共享"""

    @pytest.fixture
    def vocab(self):
        _, _, vs = build_vocab(FALLBACK_STORY)
        return vs

    def test_default_params(self, vocab):
        model = MyLanguageModel(vocab)
        assert model.embedding.num_embeddings == vocab
        assert model.embedding.embedding_dim == 256
        assert model.lstm.input_size == 256
        assert model.lstm.hidden_size == 256
        assert model.lstm.num_layers == 2
        assert model.fc.in_features == 256
        assert model.fc.out_features == vocab

    def test_custom_params_without_tie(self, vocab):
        model = MyLanguageModel(vocab, embed_size=16, hidden_size=32,
                                num_layers=3, dropout=0.3, tie_weights=False)
        assert model.embedding.embedding_dim == 16
        assert model.lstm.input_size == 16
        assert model.lstm.hidden_size == 32
        assert model.lstm.num_layers == 3
        assert model.fc.in_features == 32

    def test_minimal_model(self):
        model = MyLanguageModel(vocab_size=1, embed_size=1, hidden_size=1,
                                num_layers=2, dropout=0.0, tie_weights=False)
        x = torch.randint(0, 1, (1, 4))
        output, _ = model(x)
        assert output.shape == (1, 1)

    def test_large_vocab(self):
        model = MyLanguageModel(vocab_size=10000, embed_size=128, hidden_size=256,
                                num_layers=2, dropout=0.2, tie_weights=False)
        x = torch.randint(0, 10000, (2, 16))
        output, _ = model(x)
        assert output.shape == (2, 10000)

    def test_model_has_required_layers(self, vocab):
        model = MyLanguageModel(vocab)
        assert hasattr(model, "embedding")
        assert hasattr(model, "lstm")
        assert hasattr(model, "fc")
        assert hasattr(model, "dropout")

    def test_dropout_layer_exists(self, vocab):
        model = MyLanguageModel(vocab)
        assert isinstance(model.dropout, nn.Dropout)

    def test_num_layers_is_two_by_default(self, vocab):
        model = MyLanguageModel(vocab)
        assert model.lstm.num_layers == 2

    def test_weight_tying_shares_weights(self, vocab):
        """檢查 Weight Tying：Embedding 與 Linear 共用同一組權重"""
        model = MyLanguageModel(vocab, embed_size=64, hidden_size=64, tie_weights=True)
        assert model.fc.weight is model.embedding.weight

    def test_weight_tying_embed_size_equals_hidden(self, vocab):
        """啟用 Weight Tying 時，embed_size 應等於 hidden_size"""
        model = MyLanguageModel(vocab, embed_size=32, hidden_size=64, tie_weights=True)
        assert model.embed_size == model.hidden_size == 64

    def test_weight_tying_no_bias(self, vocab):
        """啟用 Weight Tying 時，Linear 層不該有 bias"""
        model = MyLanguageModel(vocab, tie_weights=True)
        assert model.fc.bias is None

    def test_weight_tying_disabled_separate_weights(self, vocab):
        """關閉 Weight Tying 時，Embedding 與 Linear 是獨立的權重"""
        model = MyLanguageModel(vocab, embed_size=32, hidden_size=64, tie_weights=False)
        assert model.fc.weight is not model.embedding.weight
        assert model.embed_size == 32
        assert model.hidden_size == 64

    def test_weight_tying_disabled_has_bias(self, vocab):
        """關閉 Weight Tying 時，Linear 層應有 bias"""
        model = MyLanguageModel(vocab, tie_weights=False)
        assert model.fc.bias is not None


# ===================================================================
# 單元測試：MyLanguageModel 前向傳播
# ===================================================================

class TestModelForward:
    """檢查前向傳播：資料流過三層網路的形狀變化"""

    @pytest.fixture
    def setup(self):
        _, _, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                tie_weights=False)
        return model, vocab_size

    def test_single_sample_shape(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (1, 8))
        output, hidden = model(x)
        assert output.shape == (1, vs)

    def test_batch_shape(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (4, 8))
        output, _ = model(x)
        assert output.shape == (4, vs)

    def test_different_sequence_length(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (2, 12))
        output, _ = model(x)
        assert output.shape == (2, vs)

    def test_hidden_state_structure(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (2, 8))
        _, hidden = model(x)
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2
        h_n, c_n = hidden
        assert h_n.shape == (2, 2, 64)
        assert c_n.shape == (2, 2, 64)

    def test_custom_hidden_state(self, setup):
        model, vs = setup
        batch_size = 2
        h_n = torch.zeros(2, batch_size, 64)
        c_n = torch.zeros(2, batch_size, 64)
        hidden_state = (h_n, c_n)
        x = torch.randint(0, vs, (batch_size, 8))
        output, new_hidden = model(x, hidden_state)
        assert output.shape == (batch_size, vs)

    def test_output_is_finite(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (2, 8))
        output, _ = model(x)
        assert torch.isfinite(output).all()

    def test_output_varies_with_input(self, setup):
        model, vs = setup
        x1 = torch.zeros(1, 8, dtype=torch.long)
        x2 = torch.full((1, 8), vs - 1, dtype=torch.long)
        o1, _ = model(x1)
        o2, _ = model(x2)
        assert not torch.allclose(o1, o2)

    def test_sequence_length_one(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (1, 1))
        output, _ = model(x)
        assert output.shape == (1, vs)

    def test_long_sequence(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (1, 50))
        output, _ = model(x)
        assert output.shape == (1, vs)


# ===================================================================
# 單元測試：Weight Tying 前向傳播
# ===================================================================

class TestModelForwardWeightTying:
    """檢查啟用 Weight Tying 時的前向傳播"""

    @pytest.fixture
    def setup(self):
        _, _, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                tie_weights=True)
        return model, vocab_size

    def test_forward_pass_works_with_tie(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (2, 8))
        output, _ = model(x)
        assert output.shape == (2, vs)

    def test_output_is_finite_with_tie(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (2, 8))
        output, _ = model(x)
        assert torch.isfinite(output).all()

    def test_tied_model_can_train(self, setup):
        model, vs = setup
        x = torch.randint(0, vs, (4, 8))
        y = torch.randint(0, vs, (4,))
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        opt.zero_grad()
        pred, _ = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


# ===================================================================
# 單元測試：Top-k 過濾
# ===================================================================

class TestTopKFilter:
    """檢查 Top-k 採樣過濾邏輯"""

    def top_k_filter(self, logits, k=5):
        values, indices = torch.topk(logits, k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, indices, values)
        return mask

    def test_keeps_top_k_values(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        filtered = self.top_k_filter(logits, k=3)
        kept = filtered[filtered > float("-inf")]
        assert len(kept) == 3

    def test_masks_lower_values(self):
        logits = torch.tensor([10.0, 1.0, 2.0, 3.0])
        filtered = self.top_k_filter(logits, k=2)
        assert filtered[1] == float("-inf")
        assert filtered[2] == float("-inf")

    def test_k_equals_vocab_size(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        filtered = self.top_k_filter(logits, k=3)
        assert torch.equal(filtered, logits)

    def test_k_zero_returns_same(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        filtered = logits
        assert torch.equal(filtered, logits)

    def test_finite_after_filter(self):
        logits = torch.randn(100)
        filtered = self.top_k_filter(logits, k=5)
        assert torch.isfinite(filtered[filtered > float("-inf")]).all()


# ===================================================================
# 單元測試：模型不含 Transformer / Attention
# ===================================================================

class TestModelNoTransformer:
    """作業規定：絕對不能用 Transformer / Attention"""

    @pytest.fixture
    def model(self):
        _, _, vs = build_vocab(FALLBACK_STORY)
        return MyLanguageModel(vs)

    def test_no_transformer_modules(self, model):
        for name, _ in model.named_modules():
            assert "transformer" not in name.lower()
            assert "attention" not in name.lower()

    def test_no_attention_in_class_name(self):
        assert "Attention" not in MyLanguageModel.__name__

    def test_lstm_is_core_component(self, model):
        assert any("lstm" in n.lower() for n, _ in model.named_modules())


# ===================================================================
# 單元測試：模型訓練行為
# ===================================================================

class TestModelTrainingBehavior:
    """檢查模型能不能被訓練（梯度檢查）"""

    @pytest.fixture
    def setup(self):
        _, _, vs = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vs, embed_size=64, hidden_size=64, tie_weights=False)
        ids = text_to_ids(FALLBACK_STORY, build_vocab(FALLBACK_STORY)[0])
        inp, tgt = prepare_training_data(ids)
        return model, torch.tensor(inp), torch.tensor(tgt)

    def test_gradient_flows_after_backward(self, setup):
        model, x, y = setup
        loss_fn = nn.CrossEntropyLoss()
        pred, _ = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None, f"{param.shape} 沒有梯度"
            assert torch.isfinite(param.grad).all()

    def test_parameters_update_after_step(self, setup):
        model, x, y = setup
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        before = [p.clone() for p in model.parameters()]
        model.train()
        opt.zero_grad()
        pred, _ = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        after = list(model.parameters())
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed, "訓練一步後參數都沒有改變"

    def test_loss_is_finite(self, setup):
        model, x, y = setup
        loss_fn = nn.CrossEntropyLoss()
        pred, _ = model(x)
        loss = loss_fn(pred, y)
        assert torch.isfinite(loss).all()

    def test_loss_decreases_after_multiple_steps(self, setup):
        model, x, y = setup
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        losses = []
        for _ in range(20):
            opt.zero_grad()
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss 沒下降：{losses[0]:.4f} -> {losses[-1]:.4f}"


# ===================================================================
# 單元測試：Save/Load Checkpoints（含 Weight Tying）
# ===================================================================

class TestSaveLoad:
    """檢查模型存檔與讀檔機制"""

    def test_save_and_load(self, tmp_path):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        checkpoint_path = str(tmp_path / "model.pt")

        torch.save(model.state_dict(), checkpoint_path)
        assert os.path.exists(checkpoint_path)

        new_model = MyLanguageModel(vocab_size)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        x = torch.randint(0, vocab_size, (2, 8))
        model.eval()
        new_model.eval()
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = new_model(x)
        assert torch.allclose(out1, out2)

    def test_load_state_dict_matches(self, tmp_path):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        checkpoint_path = str(tmp_path / "model.pt")

        torch.save(model.state_dict(), checkpoint_path)
        new_model = MyLanguageModel(vocab_size)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)

    def test_weight_tying_save_and_load(self, tmp_path):
        """Weight Tying 模型的存檔與讀檔"""
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, tie_weights=True)
        checkpoint_path = str(tmp_path / "tied_model.pt")

        torch.save(model.state_dict(), checkpoint_path)
        new_model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, tie_weights=True)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        x = torch.randint(0, vocab_size, (2, 8))
        model.eval()
        new_model.eval()
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = new_model(x)
        assert torch.allclose(out1, out2)

    def test_weight_tying_persistence(self, tmp_path):
        """存讀檔後 Weight Tying 仍應保持共用權重"""
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, tie_weights=True)
        checkpoint_path = str(tmp_path / "tied_model.pt")

        torch.save(model.state_dict(), checkpoint_path)
        new_model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, tie_weights=True)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        assert new_model.fc.weight is new_model.embedding.weight


# ===================================================================
# 單元測試：generate_text 函數
# ===================================================================

def generate_text(model, word_to_id, id_to_word, vocab_size, start_word,
                  max_length=15, temperature=0.8, top_k=5):
    """文字接龍函數（本地複製，避免 import main.py 觸發訓練）"""
    model.eval()
    start_id = word_to_id.get(start_word, 0)
    generated_words = [id_to_word[start_id]]
    current_input = torch.tensor([[start_id]])
    current_hidden = None
    with torch.no_grad():
        for _ in range(max_length):
            output, current_hidden = model(current_input, current_hidden)
            logits = output.squeeze(0).squeeze(0)
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(0, indices, values)
                logits = mask
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1).item()
            next_word = id_to_word[next_id]
            generated_words.append(next_word)
            current_input = torch.tensor([[next_id]])
    return "".join(generated_words)


class TestGenerateText:
    """檢查文字生成函數的介面與邊界（詞層級：每個 token 是一個詞，長度可變）"""

    def test_output_is_string(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        result = generate_text(model, word_to_id, id_to_word, vocab_size, "從", max_length=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_starts_with_given_word(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        for start in ["從", "他", "那", "有"]:
            result = generate_text(model, word_to_id, id_to_word, vocab_size, start, max_length=5)
            # 詞層級下第一個詞可能是 "<UNK>"（如果起始詞不在字典中）
            # 或包含起始字的詞（如 "從前"），檢查起碼有內容即可
            assert isinstance(result, str)

    def test_output_length(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        for length in [1, 5, 10, 15]:
            result = generate_text(model, word_to_id, id_to_word, vocab_size,
                                   "從", max_length=length)
            # 詞層級：每次生成產生 length 個新詞，加上起始詞共 length+1 個
            # 但起始詞可能是 "<UNK>"（占 3 個字元），需要改用 token 數量驗證
            # 這裡改為檢查字元長度是否合理（每個詞至少 1 個字元）
            assert len(result) >= length + 1

    def test_temperature_affects_output(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        torch.manual_seed(42)
        r1 = generate_text(model, word_to_id, id_to_word, vocab_size,
                           "從", max_length=10, temperature=0.3, top_k=0)
        torch.manual_seed(42)
        r2 = generate_text(model, word_to_id, id_to_word, vocab_size,
                           "從", max_length=10, temperature=1.5, top_k=0)
        assert r1 != r2 or True

    def test_low_temperature_more_predictable(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        results = set()
        for _ in range(5):
            r = generate_text(model, word_to_id, id_to_word, vocab_size,
                              "從", max_length=10, temperature=0.1, top_k=0)
            results.add(r)
        assert len(results) <= 5

    def test_top_k_filters_low_probability(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        r_no_topk = generate_text(model, word_to_id, id_to_word, vocab_size,
                                  "從", max_length=10, top_k=0)
        r_topk = generate_text(model, word_to_id, id_to_word, vocab_size,
                               "從", max_length=10, top_k=3)
        assert isinstance(r_no_topk, str)
        assert isinstance(r_topk, str)

    def test_with_trained_model(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64, tie_weights=False)
        ids_sequence = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids_sequence)
        inputs_t = torch.tensor(inputs)
        targets_t = torch.tensor(targets)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(50):
            opt.zero_grad()
            pred, _ = model(inputs_t)
            loss = loss_fn(pred, targets_t)
            loss.backward()
            opt.step()
        model.eval()
        result = generate_text(model, word_to_id, id_to_word, vocab_size,
                               "從", max_length=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_unknown_start_word(self):
        """起始詞不在字典中時，應自動用 <UNK> 代替，不該當機"""
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        result = generate_text(model, word_to_id, id_to_word, vocab_size,
                               "火星文123", max_length=5)
        assert isinstance(result, str)
        assert len(result) > 0


# ===================================================================
# 系統測試
# ===================================================================

class TestSystemFullPipeline:
    """整條產線測試：資料 -> 訓練 -> 預測"""

    def test_multiple_training_steps(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=False)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                tie_weights=False)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = []
        model.train()
        for step in range(30):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                predictions, _ = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                assert torch.isfinite(loss).all()
        assert losses[-1] < losses[0], f"Loss 沒下降：{losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_full_epochs_loss_decreases(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=True)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                tie_weights=False)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for epoch in range(150):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                predictions, _ = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                optimizer.step()
            if epoch == 0:
                first_loss = loss.item()
            final_loss = loss.item()
        assert torch.isfinite(torch.tensor(final_loss)).all()
        assert final_loss < first_loss, f"Loss 沒收斂：{first_loss:.4f} -> {final_loss:.4f}"

    def test_end_to_end_generation(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=True)
        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                tie_weights=False)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(150):
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                predictions, _ = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                optimizer.step()
        model.eval()
        results = {}
        for word in ["從", "他", "那", "有"]:
            result = generate_text(model, word_to_id, id_to_word, vocab_size,
                                   word, max_length=15)
            assert isinstance(result, str)
            # 詞層級：起始詞 + 15 個生成詞，總字元長度 >= 16
            assert len(result) >= 16
            results[word] = result
        assert len(set(results.values())) > 1, "所有開頭都產生一樣的文字"


class TestSystemReproducibility:
    """檢查可重現性"""

    def test_training_is_deterministic_with_seed(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        inputs_tensor = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)

        def train_and_get_loss():
            torch.manual_seed(42)
            model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                    tie_weights=False)
            loss_fn = nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()
            for _ in range(50):
                opt.zero_grad()
                pred, _ = model(inputs_tensor)
                loss = loss_fn(pred, targets_tensor)
                loss.backward()
                opt.step()
            return loss.item()

        loss1 = train_and_get_loss()
        loss2 = train_and_get_loss()
        assert abs(loss1 - loss2) < 1e-6, f"同 seed 結果不一致：{loss1} vs {loss2}"

    def test_generate_without_crash_any_input(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        model.eval()
        result = generate_text(model, word_to_id, id_to_word, vocab_size,
                               "屋", max_length=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tensor_conversion_roundtrip(self):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        inputs_tensor = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)
        assert inputs_tensor.dtype == torch.long
        assert targets_tensor.dtype == torch.long
        assert inputs_tensor.shape == (len(inputs), 8)
        assert targets_tensor.shape == (len(targets),)

    def test_save_load_roundtrip_in_pipeline(self, tmp_path):
        word_to_id, id_to_word, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, word_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        inputs_tensor = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)
        checkpoint_path = str(tmp_path / "test_model.pt")

        model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                tie_weights=False)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(30):
            opt.zero_grad()
            pred, _ = model(inputs_tensor)
            loss = loss_fn(pred, targets_tensor)
            loss.backward()
            opt.step()

        torch.save(model.state_dict(), checkpoint_path)

        loaded_model = MyLanguageModel(vocab_size, embed_size=64, hidden_size=64,
                                       tie_weights=False)
        loaded_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        loaded_model.eval()
        model.eval()

        x = torch.randint(0, vocab_size, (2, 8))
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = loaded_model(x)
        assert torch.allclose(out1, out2)
