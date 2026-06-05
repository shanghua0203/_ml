MODEL_NAME = "state-spaces/mamba-1.4b-hf"
DATASET_PATH = "data/processed/output.jsonl"
OUTPUT_DIR = "./coffee_mamba_lora"
LOG_FILE = "data/processed/training_log.jsonl"

LORA_R = 8               # r 是 LoRA 矩陣的 rank。值越高，LoRA 矩陣越大，能捕捉的資訊越多，但同時也增加參數量。8~16 是常見的起始點。
LORA_ALPHA = 16          # alpha 用來調整 LoRA 層的強度。一般建議設為 r 的兩倍。alpha 越高，LoRA 對原始模型的影響越大。
LORA_DROPOUT = 0.1      # dropout 類似 dropout，用於防止過擬合。0.05~0.1 是常見的起始點。
TARGET_MODULES = ["in_proj", "x_proj"] # in_proj：這是 Mamba 的核心層，負責輸入投影。x_proj 是 Mamba 的 another核心層，負責輸出投影。

BATCH_SIZE = 1           # 1 單純測試
GRADIENT_ACCUMULATION_STEPS = 4     # 4 一次看完 4 筆資料，再更新模型
LEARNING_RATE = 1e-4    # 這是模型在每次更新時，對模型權重的調整幅度。一般建議設為 1e-4 ~ 5e-4。
NUM_EPOCHS = 20          # 跑 20 輪
LOGGING_STEPS = 10      # 每跑 10 步記錄一次
MAX_SEQ_LENGTH = 2048   # 一次輸入的最大長度
SAVE_TOTAL_LIMIT = 10    # 最多只保留 10 個檢查點
DATALOADER_NUM_WORKERS = 2 # 2 個背景執行緒
