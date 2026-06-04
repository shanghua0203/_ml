from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling

from config import DATASET_PATH, MAX_SEQ_LENGTH


def load_dataset():
    print(f"[INFO] 載入 tokenized dataset：{DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    print(f"[INFO] 資料集筆數: {len(dataset)}")
    return dataset


def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
