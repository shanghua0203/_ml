from datasets import Dataset, load_dataset as hf_load_dataset
from transformers import DataCollatorForLanguageModeling

from config import DATASET_PATH, MAX_SEQ_LENGTH


def load_dataset(tokenizer):
    print(f"[INFO] 載入原始問答對：{DATASET_PATH}")
    raw = hf_load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"[INFO] 原始筆數: {len(raw)}")

    has_template = tokenizer.chat_template is not None

    def tokenize_fn(examples):
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            if has_template:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = f"User: {prompt}\n\nAssistant: {completion}{tokenizer.eos_token}"
            texts.append(text)

        tokenized = tokenizer(
            texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
        return tokenized

    dataset = raw.map(tokenize_fn, batched=True, batch_size=100, remove_columns=raw.column_names)
    print(f"[INFO] 分詞後資料集筆數: {len(dataset)}")
    return dataset


def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
