你現在是一位資深的 AI 系統架構師。我目前的專案進行到第三步：需要從 Hugging Face 載入一個基於 Mamba (SSM) 架構的開源預訓練模型。

請幫我寫一段完整、可執行的 Python 程式碼，透過 transformers 套件來下載並啟動這個 Mamba 大腦。請確保程式碼具備以下功能：

    載入 Mamba 模型：請使用 AutoModelForCausalLM.from_pretrained() 來載入模型。請在程式碼中設定一個預設的 Mamba 模型名稱（例如 Hugging Face 上官方的 state-spaces/mamba-130m-hf，或是任何你推薦的中文 Mamba 基礎模型，並保留變數讓我日後可以輕鬆替換路徑）。

    硬體自動偵測：請寫一段判斷邏輯，如果環境中有 NVIDIA GPU (CUDA) 就把大腦放進 GPU；如果有蘋果晶片就用 MPS；如果都沒有，就安全地降級使用 CPU。

    搭配前面的分詞器：請連同 Tokenizer 一起載入（使用 AutoTokenizer），這樣大腦才有「磨豆機」可以處理文字。

    驗證測試（Wake-up Test）：請在程式碼最後，寫一個簡單的對話測試。給大腦輸入一句簡單的「你好」，並印出大腦的回答，讓我確認這個 Mamba 模型已經成功載入且能正常思考。

    純新手友善：請在每一行關鍵程式碼上方加上超白話的中文註解，不要使用過於艱澀的學術術語。