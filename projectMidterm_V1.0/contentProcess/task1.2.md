# 角色與任務
你是一位精通 FastAPI、PyTorch 以及 HuggingFace (Transformers, PEFT) 的資深 AI 後端工程師。
請幫我修改 `scripts/main.py` 中的程式碼，解決在切換 LoRA 模型時發生的「權重污染 (Weight Pollution)」問題。

# 問題背景與現況描述
目前在 `scripts/main.py` 檔案中的 `/api/chat` 路由裡，處理 LoRA 掛載的邏輯有缺陷：
1. 一開始伺服器啟動時，載入了 `base_model`。
2. 當使用者選擇某個 LoRA 時，程式使用了 `PeftModel.from_pretrained(base_model, lora_path)`。這會直接在記憶體中修改或包裝原有的基礎模型。
3. 當使用者再次請求，並將 `req.lora_path` 設為 `null` (希望使用純基礎模型) 時，程式只是將 `active_model` 指派回 `base_model`。但此時的 `base_model` 其實已經被先前的 LoRA 權重污染了，導致前端雖然顯示切換回基礎模型，生成的內容卻依然帶有 LoRA 的特徵。
4. 即使在不同的 LoRA 之間切換 (`load_adapter`)，也無法保證權重被 100% 乾淨地替換。

# 需求：實作「完全重新載入 (Hard Reload)」機制
為了徹底避免權重污染，我希望放棄動態切換 adapter 的做法，改為**「每次切換狀態時，完全清空記憶體並重新載入模型」**。

請幫我重構 `scripts/main.py` 的全域變數與模型載入邏輯，必須滿足以下條件：
1. **加入狀態追蹤**：新增一個全域變數（例如 `current_lora_path`），用來記錄當前記憶體中正在使用的模型狀態。如果目前是純基礎模型，值為 `None`；如果是掛載了 LoRA，值為該 LoRA 的路徑。
2. **比較與觸發重載**：在 `/api/chat` 接收到請求時，比對 `req.lora_path` 與 `current_lora_path`。如果兩者相同，就繼續沿用目前的模型；**如果兩者不同，則必須觸發重新載入流程**。
3. **安全釋放記憶體**：在重新載入前，必須確保舊模型被徹底刪除。請使用 `del model`, `del base_model`，並呼叫 `import gc; gc.collect()` 以及 `torch.cuda.empty_cache()` (或對應的 mps 清理機制) 來釋放 VRAM，避免發生 Out of Memory (OOM)。
4. **乾淨的載入流程**：
   - 清理完畢後，重新使用 `AutoModelForCausalLM.from_pretrained` 載入乾淨的 Mamba 基礎模型。
   - 如果新的請求 `req.lora_path` 不為空，則使用 `PeftModel.from_pretrained` 將其掛載。
   - 更新 `current_lora_path` 為新的狀態。
5. **執行緒安全 (Thread Safety)**：因為重新載入模型需要幾秒鐘甚至更久，請考慮加入簡單的 `threading.Lock()` (例如 `model_lock`)。在檢查狀態與重新載入模型時鎖住，防止多個請求同時觸發重新載入而導致崩潰。

# 輸出要求
請直接提供修改後的 `scripts/main.py` 完整程式碼，或針對上述需求提供完整的替換區塊程式碼，並在關鍵的「記憶體清理」與「重新載入」的地方加上中文註解說明。