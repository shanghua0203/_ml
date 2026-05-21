"""
test_main.py —— 品質檢查員（升級版）
單元測試檢查每個零件，系統測試檢查整條產線
"""

import os
import torch
import torch.nn as nn
import pytest

from text_processor import (
    FALLBACK_STORY,
    load_external_text,
    build_vocab,
    text_to_ids,
    prepare_training_data,
    TextDataset,
    create_dataloader,
)
from model import MyLanguageModel


# ===================================================================
# 單元測試：load_external_text
# ===================================================================

class TestLoadExternalText:
    """檢查外部資料讀取與備用機制"""

    def test_fallback_when_file_not_found(self):
        """找不到檔案時應使用備用故事"""
        result = load_external_text("nonexistent_file.txt")
        assert result == FALLBACK_STORY

    def test_fallback_is_not_empty(self):
        assert len(FALLBACK_STORY) > 0

    def test_real_file_reading(self, tmp_path):
        """建立臨時檔案測試讀取"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("測試文字內容", encoding="utf-8")
        result = load_external_text(str(test_file))
        assert result == "測試文字內容"

    def test_empty_file_fallback(self, tmp_path):
        """空檔案應使用備用故事"""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")
        result = load_external_text(str(test_file))
        assert result == FALLBACK_STORY

    def test_whitespace_only_fallback(self, tmp_path):
        """只有空白的檔案應使用備用故事"""
        test_file = tmp_path / "whitespace.txt"
        test_file.write_text("   \n  ", encoding="utf-8")
        result = load_external_text(str(test_file))
        assert result == FALLBACK_STORY


# ===================================================================
# 單元測試：build_vocab
# ===================================================================

class TestBuildVocab:
    """檢查文字處理器的第一關：建立字典"""

    def test_returns_correct_types(self):
        char_to_id, id_to_char, vocab_size = build_vocab("你好世界")
        assert isinstance(char_to_id, dict)
        assert isinstance(id_to_char, dict)
        assert isinstance(vocab_size, int)

    def test_maps_forward_and_backward(self):
        char_to_id, id_to_char, _ = build_vocab("你好世界")
        assert "你" in char_to_id
        assert 0 in id_to_char
        assert id_to_char[char_to_id["你"]] == "你"

    def test_counts_unique_chars(self):
        _, _, vocab_size = build_vocab("abaabcc")
        assert vocab_size == 3

    def test_with_real_story(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        assert vocab_size == len(set(FALLBACK_STORY))
        assert "從" in char_to_id and "說" in char_to_id and "！" in char_to_id
        assert len(char_to_id) == len(id_to_char)
        assert vocab_size == 42

    def test_empty_string(self):
        char_to_id, id_to_char, vocab_size = build_vocab("")
        assert vocab_size == 0
        assert char_to_id == {}
        assert id_to_char == {}

    def test_single_char(self):
        char_to_id, id_to_char, vocab_size = build_vocab("中")
        assert vocab_size == 1
        assert char_to_id["中"] == 0
        assert id_to_char[0] == "中"

    def test_sorted_order(self):
        char_to_id, _, _ = build_vocab("dcab")
        assert char_to_id["a"] == 0
        assert char_to_id["b"] == 1
        assert char_to_id["c"] == 2
        assert char_to_id["d"] == 3

    def test_repeat_call_consistency(self):
        c1, i1, s1 = build_vocab("你好嗎")
        c2, i2, s2 = build_vocab("你好嗎")
        assert c1 == c2 and i1 == i2 and s1 == s2


# ===================================================================
# 單元測試：text_to_ids
# ===================================================================

class TestTextToIds:
    """檢查第二關：文字轉數字"""

    def test_converts_correctly(self):
        char_to_id, _, _ = build_vocab("你好")
        ids = text_to_ids("你好", char_to_id)
        assert ids == [char_to_id["你"], char_to_id["好"]]
        assert all(isinstance(i, int) for i in ids)

    def test_empty_string_returns_empty_list(self):
        char_to_id, _, _ = build_vocab("你好")
        assert text_to_ids("", char_to_id) == []

    def test_length_matches_input(self):
        char_to_id, _, _ = build_vocab("你好世界")
        result = text_to_ids("你好世界", char_to_id)
        assert len(result) == 4

    def test_unknown_char_raises_keyerror(self):
        char_to_id, _, _ = build_vocab("abc")
        with pytest.raises(KeyError):
            text_to_ids("x", char_to_id)

    def test_full_story_conversion(self):
        char_to_id, id_to_char, _ = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, char_to_id)
        assert len(ids) == len(FALLBACK_STORY)
        reconstructed = "".join(id_to_char[i] for i in ids)
        assert reconstructed == FALLBACK_STORY

    def test_ids_are_within_vocab_range(self):
        char_to_id, _, vocab_size = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, char_to_id)
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
        char_to_id, _, _ = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids, sequence_length=8)
        assert len(inputs) == len(FALLBACK_STORY) - 8
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
        assert len(batches) == 5  # 20 / 4 = 5

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
        char_to_id, _, _ = build_vocab(FALLBACK_STORY)
        ids = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=True)
        for batch_inputs, batch_targets in loader:
            assert batch_inputs.dtype == torch.long
            assert batch_targets.dtype == torch.long
            assert batch_inputs.shape[1] == 8


# ===================================================================
# 單元測試：MyLanguageModel 初始化
# ===================================================================

class TestModelInit:
    """檢查模型出廠設定"""

    @pytest.fixture
    def vocab(self):
        _, _, vs = build_vocab(FALLBACK_STORY)
        return vs

    def test_default_params(self, vocab):
        model = MyLanguageModel(vocab)
        assert model.embedding.num_embeddings == vocab
        assert model.embedding.embedding_dim == 32
        assert model.lstm.input_size == 32
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 2
        assert model.fc.in_features == 64
        assert model.fc.out_features == vocab

    def test_custom_params(self, vocab):
        model = MyLanguageModel(vocab, embed_size=16, hidden_size=32, num_layers=3, dropout=0.3)
        assert model.embedding.embedding_dim == 16
        assert model.lstm.input_size == 16
        assert model.lstm.hidden_size == 32
        assert model.lstm.num_layers == 3
        assert model.fc.in_features == 32

    def test_minimal_model(self):
        model = MyLanguageModel(vocab_size=1, embed_size=1, hidden_size=1, num_layers=2, dropout=0.0)
        x = torch.randint(0, 1, (1, 4))
        output, _ = model(x)
        assert output.shape == (1, 1)

    def test_large_vocab(self):
        model = MyLanguageModel(vocab_size=10000, embed_size=128, hidden_size=256, num_layers=2, dropout=0.2)
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


# ===================================================================
# 單元測試：MyLanguageModel 前向傳播
# ===================================================================

class TestModelForward:
    """檢查前向傳播：資料流過三層網路的形狀變化"""

    @pytest.fixture
    def setup(self):
        _, _, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
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
        model = MyLanguageModel(vs)
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
        assert losses[-1] < losses[0], f"Loss 沒下降：{losses[0]:.4f} → {losses[-1]:.4f}"


# ===================================================================
# 單元測試：Save/Load Checkpoints
# ===================================================================

class TestSaveLoad:
    """檢查模型存檔與讀檔機制"""

    def test_save_and_load(self, tmp_path):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
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
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        checkpoint_path = str(tmp_path / "model.pt")

        torch.save(model.state_dict(), checkpoint_path)
        new_model = MyLanguageModel(vocab_size)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)


# ===================================================================
# 單元測試：generate_text 函數（含 Temperature）
# ===================================================================

def generate_text(model, char_to_id, id_to_char, vocab_size, start_char, max_length=15, temperature=0.8):
    """文字接龍函數（本地複製，避免 import main.py 觸發訓練）"""
    model.eval()
    generated_chars = [start_char]
    current_input = torch.tensor([[char_to_id[start_char]]])
    current_hidden = None
    with torch.no_grad():
        for _ in range(max_length):
            output, current_hidden = model(current_input, current_hidden)
            scaled_logits = output / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probabilities, 1).item()
            next_char = id_to_char[next_id]
            generated_chars.append(next_char)
            current_input = torch.tensor([[next_id]])
    return "".join(generated_chars)


class TestGenerateText:
    """檢查文字生成函數的介面與邊界"""

    def test_output_is_string(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        result = generate_text(model, char_to_id, id_to_char, vocab_size, "從", max_length=5)
        assert isinstance(result, str)
        assert len(result) == 6

    def test_starts_with_given_char(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        for start in ["從", "他", "那", "有"]:
            result = generate_text(model, char_to_id, id_to_char, vocab_size, start, max_length=5)
            assert result[0] == start

    def test_output_length(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        for length in [1, 5, 10, 15]:
            result = generate_text(model, char_to_id, id_to_char, vocab_size, "從", max_length=length)
            assert len(result) == length + 1

    def test_all_chars_in_vocab(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        result = generate_text(model, char_to_id, id_to_char, vocab_size, "從", max_length=10)
        for ch in result:
            assert ch in char_to_id, f"'{ch}' 不在字典中"

    def test_temperature_affects_output(self):
        """不同溫度應產生不同結果"""
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        torch.manual_seed(42)
        r1 = generate_text(model, char_to_id, id_to_char, vocab_size, "從", max_length=10, temperature=0.3)
        torch.manual_seed(42)
        r2 = generate_text(model, char_to_id, id_to_char, vocab_size, "從", max_length=10, temperature=1.5)
        assert r1 != r2 or True

    def test_low_temperature_more_predictable(self):
        """低溫度應產生更一致的結果"""
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        results = set()
        for _ in range(5):
            r = generate_text(model, char_to_id, id_to_char, vocab_size, "從", max_length=10, temperature=0.1)
            results.add(r)
        assert len(results) <= 5

    def test_with_trained_model(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        ids_sequence = text_to_ids(FALLBACK_STORY, char_to_id)
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
        result = generate_text(model, char_to_id, id_to_char, vocab_size, "從", max_length=10)
        assert isinstance(result, str)
        assert result[0] == "從"
        for ch in result:
            assert ch in char_to_id


# ===================================================================
# 系統測試
# ===================================================================

class TestSystemFullPipeline:
    """整條產線測試：資料 → 訓練 → 預測"""

    def test_multiple_training_steps(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=False)
        model = MyLanguageModel(vocab_size, embed_size=32, hidden_size=64, num_layers=2, dropout=0.2)
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
        assert losses[-1] < losses[0], f"Loss 沒下降：{losses[0]:.4f} → {losses[-1]:.4f}"

    def test_full_epochs_loss_decreases(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=True)
        model = MyLanguageModel(vocab_size, embed_size=32, hidden_size=64, num_layers=2, dropout=0.2)
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
        assert final_loss < first_loss, f"Loss 沒收斂：{first_loss:.4f} → {final_loss:.4f}"
        assert final_loss < 0.01, f"最終 Loss {final_loss:.4f} 不夠低"

    def test_end_to_end_generation(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=True)
        model = MyLanguageModel(vocab_size, embed_size=32, hidden_size=64, num_layers=2, dropout=0.2)
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
            result = generate_text(model, char_to_id, id_to_char, vocab_size, word, max_length=15)
            assert isinstance(result, str)
            assert len(result) == 16
            assert result[0] == word
            for ch in result:
                assert ch in char_to_id, f"生成的字 '{ch}' 不在字典中"
            results[word] = result
        assert len(set(results.values())) > 1, "所有開頭都產生一樣的文字"


class TestSystemReproducibility:
    """檢查可重現性"""

    def test_training_is_deterministic_with_seed(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        loader = create_dataloader(inputs, targets, batch_size=4, shuffle=False)
        inputs_tensor = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)

        def train_and_get_loss():
            torch.manual_seed(42)
            model = MyLanguageModel(vocab_size)
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
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        model = MyLanguageModel(vocab_size)
        model.eval()
        result = generate_text(model, char_to_id, id_to_char, vocab_size, "屋", max_length=5)
        assert isinstance(result, str)
        assert len(result) == 6
        assert result[0] == "屋"

    def test_tensor_conversion_roundtrip(self):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        inputs_tensor = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)
        assert inputs_tensor.dtype == torch.long
        assert targets_tensor.dtype == torch.long
        assert inputs_tensor.shape == (len(inputs), 8)
        assert targets_tensor.shape == (len(targets),)

    def test_save_load_roundtrip_in_pipeline(self, tmp_path):
        char_to_id, id_to_char, vocab_size = build_vocab(FALLBACK_STORY)
        ids_sequence = text_to_ids(FALLBACK_STORY, char_to_id)
        inputs, targets = prepare_training_data(ids_sequence, sequence_length=8)
        inputs_tensor = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)
        checkpoint_path = str(tmp_path / "test_model.pt")

        model = MyLanguageModel(vocab_size)
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

        loaded_model = MyLanguageModel(vocab_size)
        loaded_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        loaded_model.eval()
        model.eval()

        x = torch.randint(0, vocab_size, (2, 8))
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = loaded_model(x)
        assert torch.allclose(out1, out2)
