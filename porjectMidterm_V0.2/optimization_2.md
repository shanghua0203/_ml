# Role & Context
You are a Senior NLP PyTorch Engineer. We are debugging a Word-level LSTM Language Model.
Current Status: The dataset contains only about 70,000 characters. After `jieba` tokenization, the vocabulary size is around 5,358. Due to the long-tail distribution of natural language (Zipf's Law) in such a micro-dataset, we are facing severe **Data Sparsity**. Many tokens appear only 1 or 2 times, causing the model to memorize noise (train_loss drops to 2.0) but fail to generalize (val_loss is stuck at 5.0).

# Objective: Vocabulary Truncation (方案 A)
We need to implement a low-frequency word filtering mechanism to reduce the vocabulary size and force the model to learn high-frequency core syntactic patterns.

# Actionable Tasks
1. **Refactor `text_processor.py`**:
   - Modify the `build_vocab` function to accept a new parameter `min_freq` (default set to 3).
   - Use `collections.Counter` to calculate token frequencies.
   - Truncate the vocabulary by discarding any token whose frequency is strictly less than `min_freq`.
   - Ensure the `UNK_TOKEN` (ID: 0) mechanism remains intact. Any discarded low-frequency token MUST naturally route to `UNK_ID` during the `text_to_ids` mapping phase.

2. **Update `main.py`**:
   - Expose `MIN_FREQ = 3` as a global hyperparameter at the top of the file.
   - Pass this hyperparameter into the `build_vocab` function call.
   - Ensure the `VOCAB_PATH` JSON dump reflects the truncated vocabulary accurately.

3. **Write Unit Tests**:
   - Update or create a `test_text_processor.py` using `pytest`.
   - Write a specific test case to verify that words appearing below the `min_freq` threshold are indeed excluded from the returned `word_to_id` dictionary and are correctly converted to `UNK_ID` (0) by the `text_to_ids` function.

# Strict Constraints
- **Performance**: Use efficient Python built-ins (`collections.Counter`) for frequency counting. Do not use unoptimized nested loops.
- **Language Requirements**: ALL code comments, docstrings, console output, and explanations MUST be written in Traditional Chinese (繁體中文). Do NOT use Simplified Chinese.
- Provide the exact refactored code blocks for `text_processor.py`, the updated portions of `main.py`, and the complete `test_text_processor.py`.