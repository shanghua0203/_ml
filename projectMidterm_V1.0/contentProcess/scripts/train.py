import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import Trainer, TrainingArguments, TrainerCallback

from config import (
    OUTPUT_DIR, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, NUM_EPOCHS, LOGGING_STEPS,
    SAVE_TOTAL_LIMIT, DATALOADER_NUM_WORKERS, LOG_FILE,
)
from model_utils import load_base_model, load_tokenizer, apply_lora, get_device
from data_utils import load_dataset, create_data_collator


class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        if loss is None:
            return
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4),
            "loss": round(loss, 6),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("=" * 50)
    print("  Mamba LoRA Fine-tuning 啟動")
    print("=" * 50)

    get_device()
    tokenizer = load_tokenizer()
    model = load_base_model()
    model = apply_lora(model)
    dataset = load_dataset()
    data_collator = create_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        report_to="none",
    )

    loss_logger = LossLoggerCallback(LOG_FILE)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[loss_logger],
    )

    print(f"\n[INFO] 開始訓練 ...（損失值同步記錄至 {LOG_FILE}）")
    trainer.train()

    save_path = OUTPUT_DIR
    print(f"\n[INFO] 儲存 LoRA 權重至 {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("[完成] LoRA 微調完成！")


if __name__ == "__main__":
    main()
