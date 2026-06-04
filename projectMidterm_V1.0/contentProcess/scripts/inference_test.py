import torch

from model_utils import load_base_model, load_tokenizer, get_device
from config import MODEL_NAME


def run_inference_test():
    print("=" * 50)
    print("  推理測試 — 確認 Mamba 模型能正常生成文字")
    print("=" * 50)

    device = get_device()
    tokenizer = load_tokenizer()
    model = load_base_model()
    model.to(device)
    model.eval()

    test_prompts = [
        "你好",
        "手沖咖啡需要哪些器具？",
        "什麼是 Mamba 模型？",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n輸入: {prompt}")
        print(f"輸出: {response}")

    print("\n[PASS] 推理測試完成 — 模型可正常生成文字")


if __name__ == "__main__":
    run_inference_test()
