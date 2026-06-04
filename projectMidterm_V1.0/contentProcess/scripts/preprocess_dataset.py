import json
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

INPUT_FILE = "data/processed/output.jsonl"
OUTPUT_DIR = "data/processed/tokenized_dataset"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def process_batch(examples):
    texts = []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    tokenized = tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding=False
    )
    tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
    return tokenized


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"[ERROR] 找不到輸入檔案: {INPUT_FILE}")
        sys.exit(1)

    print(f"[INFO] 讀取 {INPUT_FILE} ...")
    dataset = load_dataset("json", data_files=str(input_path), split="train")
    print(f"[INFO] 原始資料筆數: {len(dataset)}")

    print("[INFO] 進行對話模板套用與分詞 ...")
    tokenized_ds = dataset.map(
        process_batch,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    output_path = Path(OUTPUT_DIR)
    print(f"[INFO] 儲存至 {OUTPUT_DIR} ...")
    tokenized_ds.save_to_disk(str(output_path))

    lengths = tokenized_ds["length"]
    avg_tokens = sum(lengths) / len(lengths)
    max_tokens = max(lengths)
    total_tokens = sum(lengths)

    print("\n" + "=" * 50)
    print("              資料統計報告")
    print("=" * 50)
    print(f"  總問答筆數:          {len(tokenized_ds)}")
    print(f"  每筆平均 Token 數:   {avg_tokens:.1f}")
    print(f"  每筆最多 Token 數:   {max_tokens}")
    print(f"  總 Token 數:         {total_tokens}")
    print("=" * 50)
    print(f"[完成] 輸出目錄: {output_path.resolve()}")


if __name__ == "__main__":
    main()
