# Role & Context
You are a Senior PyTorch Engineer specializing in NLP. We are debugging and optimizing a custom PyTorch LSTM Language Model (Word-level, using `jieba`).

# Current Status & Diagnostics
Based on the recent `training_log.txt`, the model has successfully overcome the previous underfitting issue but is now suffering from SEVERE OVERFITTING. 
- The `train_loss` successfully decreased from 6.2191 down to 0.0540, proving the model has the capacity to learn.
- However, the `val_loss` hit its absolute minimum at Epoch 3 (5.3018) and then consistently degraded, reaching 10.8857 by Epoch 115.
- Due to the degrading validation metric, the LR scheduler decayed the learning rate from 0.0005 to an effectively zero value of 4.9e-7, leading to useless computational cycles.

# Strict Constraints
1. **NO TRANSFORMERS**: The use of Multi-Head Attention, Transformer layers, or any self-attention mechanism is strictly prohibited.
2. **Framework**: Native PyTorch only.
3. **Language Requirement**: ALL code comments, docstrings, console outputs, and documentation MUST be written in Traditional Chinese (繁體中文).

# Actionable Tasks
Please refactor `main.py` and `model.py` focusing strictly on mitigating overfitting and preventing wasted compute time. Implement the following strategies:

1. **Enhance Regularization**:
   - The current `dropout=0.1` is insufficient for a `hidden_size=256`[cite: 2]. Increase the `dropout` parameter in the LSTM layer to a value between `0.3` and `0.5`.
   - Introduce L2 Regularization by adding `weight_decay` (e.g., `1e-4` or `1e-5`) to the Adam optimizer instantiation.

2. **Implement Robust Early Stopping & Checkpoint Restoration**:
   - Implement an Early Stopping logic that monitors `val_loss`.
   - Halt the training loop if `val_loss` does not improve for a defined `patience` threshold (e.g., 7 to 10 epochs).
   - **CRITICAL**: The script MUST automatically save the `state_dict` when a new best `val_loss` is found, and RELOAD these best weights immediately after Early Stopping is triggered, discarding the overfitted weights.

3. **Model Capacity Tuning**:
   - Given the `vocab_size=5358`[cite: 2], a parameter size of 256 might be over-parameterized for the current dataset volume. Propose a toggle or updated configuration to scale down both `embed_size` and `hidden_size` to `128` as a fallback experiment to find the optimal bias-variance tradeoff.

Please provide the exact refactored Python code blocks for the optimizer setup, the Early Stopping implementation within the training loop, and the updated model instantiation.