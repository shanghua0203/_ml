import torch
from transformers import AutoTokenizer, MambaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from config import MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES


def load_base_model():
    print(f"[INFO] 載入 Mamba 模型：{MODEL_NAME}")
    model = MambaForCausalLM.from_pretrained(MODEL_NAME)
    model.train()
    return model


def load_tokenizer():
    print(f"[INFO] 載入分詞器：{MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_lora(model):
    print("[INFO] 套用 LoRA ...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，無法進行訓練。請確認 GPU 驅動與 PyTorch 版本。")
    device = torch.device("cuda")
    print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}")
    return device
