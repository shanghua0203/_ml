你現在是一位資深的 Python AI 系統工程師。我需要一段完整且可執行的 Python 腳本，用來將一份約 7 萬字的純文字檔（input.txt），透過呼叫本地端的 Ollama（模型：gemma3:4b），自動轉換成問答對，並存成 JSONL 格式的檔案（output.jsonl）。

請確保你的程式碼包含以下關鍵機制：

    文本切割（Chunking）：因為 7 萬字會超出 gemma3:4b 的 Context Window，請幫我寫一個簡單的分塊函數（例如每次讀取 500 到 1000 字），最好是能盡量以「段落」或「句號」來切斷，避免語意破裂。

    呼叫 Ollama API：請使用 Python 的 requests 套件呼叫 http://localhost:11434/api/generate，或者使用官方的 ollama python 套件。

    強制 JSON 輸出：在程式碼裡設計給 gemma3:4b 的 System Prompt，嚴格限制模型只能輸出這樣的格式：{"prompt": "整理出的問題寫這", "completion": "答案寫這"}，絕對不能包含任何 Markdown 標記（如 ```json）或其他廢話。可以使用模型的 format="json" 參數（若適用）。

    容錯與除錯機制（Error Handling）：AI 有時候會吐出壞掉的 JSON。請務必加上 try-except 與 json.loads 來驗證。如果解析失敗，印出警告並跳過該區塊，程式絕對不能因此中斷。

    即時寫入（Append Mode）：為了怕跑到一半斷電，請確保每成功轉換一組 QA，就立刻以 a (append) 模式寫入 output.jsonl 檔案中，並附帶簡單的進度條（如 tqdm）或 print 提示。