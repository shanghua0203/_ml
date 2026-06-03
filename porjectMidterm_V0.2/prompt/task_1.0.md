你現在是一位資深 PyTorch 軟體工程師，且擅長用極度白話、連國小生都能懂的比喻來寫程式碼註解。
我目前有一個「非 Transformer 的 LSTM 中文語言模型」專案，包含 text_processor.py, model.py, main.py 三個檔案。

目前專案只是個雛形，請幫我針對這三個檔案進行「專業化升級」，請給我修改後的完整程式碼。
⚠️ 嚴格限制：絕對禁止使用 Transformer、Attention 或是 Multi-Head Attention 機制。

請幫我實作以下 5 個升級功能：

1. 實作「存檔與讀檔」機制 (Save/Load Checkpoints)
- 在 main.py 訓練迴圈結束後，使用 torch.save 儲存模型的 state_dict。
- 在文字生成前，加上可以讀取權重 (torch.load) 的邏輯，讓模型不用每次都重新訓練。

2. 改用 DataLoader 處理資料 (Batching)
- 在 text_processor.py 中，引入 torch.utils.data 的 Dataset 與 DataLoader。
- 將原本一次塞入全部資料的寫法，改成可以使用 batch_size 批次訓練的推車模式，並在 main.py 套用。

3. 支援讀取外部 .txt 檔案 (External Dataset)
- 在 text_processor.py 寫一個可以讀取外部文本（例如 dataset.txt）的函數。
- 如果找不到 txt 檔案，才使用原本的 50 字假故事當作備用資料 (Fallback)。

4. 加入 Dropout 防止過度擬合 (Overfitting)
- 在 model.py 的 LSTM 層中設定 dropout 參數（例如 dropout=0.2），讓模型在訓練時隨機遺忘，學會真正理解上下文而不是死背。
- 注意：如果要加 dropout，LSTM 的 num_layers 必須大於 1，請幫我把層數改為 2。

5. 實作 Temperature 採樣機制
- 在 main.py 的 generate_text 函數中，加入 temperature 參數（預設 0.8）。
- 在 softmax 之前，將模型的輸出 logits 除以 temperature，藉此控制模型生成文字的「隨機度與創意度」。

【註解要求】
請在每一段修改過的程式碼旁邊，加上「超級白話」的中文註解。例如：把 DataLoader 比喻成手推車、把 Dropout 比喻成防止死背的機制、把 Temperature 比喻成控制模型講話創意的溫度計。
