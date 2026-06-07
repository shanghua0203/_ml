# 任務一：Mamba 模型推理腳本

請幫我撰寫一個 Python 腳本，用於執行 Hugging Face 模型的文字生成推理（Inference）。
我使用的是 Mamba 架構的模型，並且已經訓練好了 LoRA 權重。目前的腳本只能載入基礎模型，請幫我升級它。

請用 argparse 設計成可以透過命令列傳遞參數的程式，並具備以下功能：

1. 基礎模型與 LoRA 載入設定：
   - 提供參數讓我可以選擇「要不要調用 LoRA 權重」（例如 --use_lora）。
   - 提供參數讓我指定「LoRA 權重的資料夾路徑」（例如 --lora_path），這在我有好幾個不同實驗版本的權重時可以方便切換。
   - 程式內部需要使用 `peft` 函式庫的 `PeftModel.from_pretrained` 將 LoRA 權重與基礎模型合併。

2. 靈活的生成參數調整（Generation Parameters）：
   - 溫度設定（--temperature）：預設 0.7。
   - 最大生成字數（--max_tokens）：預設 200，避免話沒說完。
   - Top-p（--top_p）：預設 0.9，用來過濾太奇怪的詞彙。
   - 重複懲罰（--repetition_penalty）：預設 1.1，防止模型像跳針一樣重複同一個字。

3. 使用者體驗功能：
   - 自動硬體偵測：如果有 CUDA 就用 GPU，有 Apple MPS 就用 MPS，都沒有才用 CPU。
   - 互動聊天模式（Interactive Mode）：啟動後，請給一個類似聊天室的輸入框（例如 `User: `），讓我可以一直輸入問題測試（例如問「手沖咖啡的參數建議？」），直到我輸入 `exit` 或 `quit` 才結束程式，不要跑一次對話就關閉。

程式碼請加上詳細且白話的中文註解，讓我這個初學者也能看懂每一行在做什麼。