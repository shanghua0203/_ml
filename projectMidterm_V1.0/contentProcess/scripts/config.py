MODEL_NAME = "state-spaces/mamba-1.4b-hf"
DATASET_PATH = "data/processed/tokenized_dataset"
OUTPUT_DIR = "./coffee_mamba_lora"
LOG_FILE = "data/processed/training_log.jsonl"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["in_proj", "x_proj"]

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LOGGING_STEPS = 10
MAX_SEQ_LENGTH = 2048
SAVE_TOTAL_LIMIT = 2
DATALOADER_NUM_WORKERS = 2
