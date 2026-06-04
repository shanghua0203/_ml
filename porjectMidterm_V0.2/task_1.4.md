# Role & Context
You are a Senior NLP PyTorch Engineer. 
Current Status: Our Word-level LSTM Language Model has successfully reduced `val_loss` to 4.04 by implementing low-frequency word filtering (`min_freq=3`). However, the inference/generation output suffers from three severe NLP issues:
1. **The `<UNK>` Black Hole**: The model over-predicts `<UNK>` tokens due to their high frequency in the truncated training data.
2. **Repetition Loops**: The greedy decoding nature of the LSTM causes it to fall into repetitive loops (e.g., "再再再").
3. **Domain Token Fragmentation**: `jieba` is inappropriately splitting highly specific domain terms (e.g., coffee terminology) into meaningless sub-tokens.

# Objective
Refactor the inference logic in `main.py` and the tokenization logic in `text_processor.py` to fix these generation issues without retraining the model.

# Actionable Tasks

1. **Implement Logit Masking for `<UNK>` (`main.py` -> `generate_text`)**:
   - Before applying `top_k_filter` and `softmax`, explicitly fetch the ID for `<UNK>`.
   - Force the logit of the `<UNK>` token to `-inf` to completely mask it out of the probability distribution.

2. **Implement Repetition Penalty (`main.py` -> `generate_text`)**:
   - Maintain a list or set of `generated_ids` during the generation loop.
   - Introduce a `repetition_penalty` hyperparameter (e.g., `1.2`).
   - Iterate through the unique IDs in `generated_ids`. If the logit for a generated ID is positive, divide it by the penalty. If it is negative, multiply it by the penalty. Apply this BEFORE `top_k_filter`.

3. **Protect Domain Dictionary (`text_processor.py` -> `tokenize`)**:
   - Before calling `jieba.lcut()`, programmatically add a list of domain-specific keywords to the jieba dictionary using `jieba.add_word()`.
   - Example domain words to protect: `["手沖", "精品咖啡", "萃取率", "水洗", "日曬", "淺焙", "中焙", "深焙", "注水", "悶蒸", "二氧化碳浸漬", "藝伎", "九十加"]`.

# Strict Constraints
- **Do NOT retrain**: These fixes must apply only to the preprocessing and inference phases.
- **Language Requirements**: ALL code comments, docstrings, console outputs, and explanations MUST be written in Traditional Chinese (繁體中文). Do NOT use Simplified Chinese.
- Provide the exact refactored code blocks for the `tokenize` function in `text_processor.py` and the `generate_text` function in `main.py`.