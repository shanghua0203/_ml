import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import torch


def test_load_base_model():
    from model_utils import load_base_model
    model = load_base_model()
    assert model is not None
    assert hasattr(model, "generate")
    del model
    torch.cuda.empty_cache()


def test_load_tokenizer():
    from model_utils import load_tokenizer
    tokenizer = load_tokenizer()
    assert tokenizer is not None
    assert tokenizer.eos_token is not None
    assert tokenizer.pad_token is not None


def test_apply_lora():
    from model_utils import load_base_model, apply_lora
    model = load_base_model()
    peft_model = apply_lora(model)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert trainable > 0, "LoRA 未產生任何可訓練參數"
    print(f"  LoRA 可訓練參數: {trainable:,}")
    del peft_model
    torch.cuda.empty_cache()


def test_get_device():
    from model_utils import get_device
    device = get_device()
    assert device.type == "cuda"


def test_tokenizer_encodes_chinese():
    from model_utils import load_tokenizer
    tokenizer = load_tokenizer()
    ids = tokenizer.encode("你好，世界！")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
