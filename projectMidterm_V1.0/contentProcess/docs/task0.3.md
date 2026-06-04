你現在 suicide 是一位頂尖的 LLM 訓練專家，專精於使用 PyTorch 與 Hugging Face 框架進行高效微調。

我現在要進行專案的第四步：使用 LoRA 技術，將第二步預處理好的 tokenized_dataset 資料集，拿來微調（Fine-tune）第三步選定的 state-spaces/mamba-1.4b-hf 模型。

請幫我寫一段完整、結構化且包含詳細中文註解的 Python 訓練腳本（train.py）。程式碼必須符合以下專業規範：

    環境與硬體優化：

        必須強制偵測並使用 CUDA（GPU）進行訓練。

        引入 peft 套件的 LoraConfig 與 get_peft_model。針對 Mamba 架構，請將 LoRA 的 target_modules 針對 Mamba 的核心投影層（例如 in_proj, out_proj，或保留註解說明如何根據模型架構調整）。

        設定 LoRA 參數：r=8, lora_alpha=16, lora_dropout=0.05。

    載入資料與模型：

        使用 datasets.load_from_disk("tokenized_dataset") 讀取資料。

        載入 state-spaces/mamba-1.4b-hf 模型，並將其包裝為 Peft/LoRA 模型。

    訓練參數設定（TrainingArguments）：

        使用 Hugging Face 的 Trainer（或如果是 Mamba 官方的 MambaTrainer 請做對應相容調整，若無則使用標準 Trainer）。

        訓練參數請優化以防顯存（VRAM）溢出（OOM）：設定 per_device_train_batch_size=1 或 2，啟用梯度累積 gradient_accumulation_steps=4。

        設定學習率 learning_rate=2e-4，訓練輪數 num_train_epochs=3。

        啟用 logging_steps=10，每 10 步印出當前的 Loss（損失值），方便我觀察 AI 有沒有越學越聰明。

    模型儲存：

        訓練完成後，將訓練好的 LoRA 外掛權重自動儲存到 ./coffee_mamba_lora 資料夾。

    純新手友善：

        請在程式碼中加上豐富的中文註解，解釋每個參數（如 r, alpha, batch_size）代表的白話意義，並在程式開始與結束時 print 提示文字。