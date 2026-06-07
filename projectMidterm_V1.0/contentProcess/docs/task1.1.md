我需要開發一個「前後端分離」的本機大語言模型 (LLM) 聊天網頁應用程式。
我使用的是 Mamba 架構模型，並使用 Hugging Face `peft` 進行了 LoRA 微調。

請幫我撰寫完整的專案程式碼，包含後端 (FastAPI) 與前端 (單一 HTML 檔 + 原生 JS)，並滿足以下所有詳細規格：

### 一、 後端需求 (FastAPI + Python)
1. 基礎架構與模型載入：
   - 使用 `FastAPI` 與 `uvicorn` 作為伺服器。
   - 在伺服器啟動時，將 Mamba 基礎模型載入到 GPU/MPS/CPU 記憶體中（保持全局常駐，不要每次請求都重新載入）。
   
2. LoRA 權重掃描與資訊解析 (API: `/api/lora_models`)：
   - 撰寫一個函式掃描專案目錄下的 `coffee_mamba_lora/` 資料夾。
   - 找出所有前綴為 `checkpoint-` 的子資料夾。
   - 嘗試讀取該資料夾內的 `trainer_state.json`，提取該 checkpoint 的 `global_step` 與最新的 `loss` 值。
   - 回傳給前端一個列表，格式包含：資料夾名稱、步數 (Step)、Loss 值（若無檔案則給預設提示）。

3. 串流對話生成 (API: `/api/chat` - Streaming Endpoint)：
   - 接收前端傳來的 JSON：`{"prompt": "...", "temperature": 0.7, "lora_path": "coffee_mamba_lora/checkpoint-100"}`。
   - 動態切換 LoRA：收到請求時，請使用 `peft` 的機制 (例如 `model.load_adapter()` 或 `model.set_adapter()`)，將使用者指定的 LoRA 權重掛載到基礎模型上，確保切換速度快。
   - 生成文字：使用 `StreamingResponse` (Server-Sent Events, SSE) 或是 yield chunk 的方式，實現文字的「打字機效果」(Streaming)。
   - 備註尾綴：在模型生成完所有內容後，請在最後 yield 出一段固定格式的文字：`\n\n> 💡 系統備註：本回答產生於 溫度 {temperature}，使用權重 {lora_folder_name}`。

### 二、 前端需求 (HTML + Vanilla JS + CSS)
1. 介面設計 (寫在同一個 index.html 中，並由 FastAPI 提供靜態檔或直接開啟)：
   - 畫面分為左側「控制面板」與右側「聊天視窗」。
   - 【控制面板】：
     - 下拉式選單：載入畫面時自動打 `/api/lora_models` API 獲取選項。選項文字需顯示為 `checkpoint-xxxx (Step: xxx, Loss: x.xx)`。
     - 溫度拉桿 (Range Slider)：範圍 0.1 到 1.5，預設 0.7，旁邊要顯示目前的數值，且能即時拉動調整。
   - 【聊天視窗】：
     - 訊息展示區：區分使用者與 AI 的對話氣泡。
     - 輸入框與發送按鈕。

2. JS 串流接收邏輯：
   - 當使用者點擊發送，獲取輸入框文字、當前選中的 LoRA 路徑、當前的溫度數值。
   - 呼叫 `/api/chat`，使用 `fetch` 搭配 `response.body.getReader()` 來讀取串流數據。
   - 讀取時，將文字一個字一個字附加到畫面上最新的 AI 對話氣泡中（打字機效果）。

請給我完整的目錄結構建議，以及 `main.py` 和 `index.html` 的完整可執行程式碼。程式碼中請加入詳細的繁體中文註解，解釋前後端串接的邏輯以及模型動態切換的寫法。