# 你現在是一位專精於大語言模型（LLM）資料預處理的資深資料科學家。
我手上有一個已經整理好的問答對檔案 output.jsonl，格式如下：
{"prompt": "問題內容", "completion": "答案內容"}

我接下來要使用這些資料來為我的本地模型進行微調（Fine-tuning）。請為我寫一段完整、乾淨、且包含中文註解的 Python 腳本，來執行資料的分詞（Tokenization）與預處理。

請確保程式碼具備以下專業功能與架構：

    載入分詞器（Tokenizer）：使用 Hugging Face 的 transformers 套件，透過 AutoTokenizer.from_pretrained() 載入指定的分詞器。請在程式碼中留一個變數當作模型名稱的預設值（例如："Qwen/Qwen2.5-7B-Instruct" 或是開放讓使用者填入本地路徑）。

    套用對話模板（Chat Template）：程式必須將 prompt 與 completion 轉換成標準的對話結構。請使用分詞器自帶的 apply_chat_template 功能，將資料包裝成如下結構：

        User: [prompt]

        Assistant: [completion]
        並且務必加上結尾符號（<eos> 或 eos_token），讓模型知道這題在這裡結束。

    分詞與序列化：將轉換後的對話文本丟進 Tokenizer 轉換成 input_ids 和 attention_mask。

    使用 Hugging Face datasets 套件管理：

        請用 datasets 套件讀取 output.jsonl。

        使用 .map() 函數高效地對整個資料集進行分詞處理。

        處理完畢後，使用 .save_to_disk("tokenized_dataset") 將結果儲存成一個乾淨、準備好可以丟給訓練框架（如 LLaMA-Factory 或 Hugging Face Trainer）的資料夾。

    列印資料統計報告：程式執行完畢後，請幫我算一下並印出：

        總共處理了幾筆問答？

        這批資料平均一組問答含有多少個 Token？最高是多少？（這能幫我評估訓練時的總 Token 數與成本）。

輸出完整可執行的 Python 程式碼，加上完整的中文說明。程式碼與中文說明
寫詳細的README.md