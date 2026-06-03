# Role & Context
You are a Senior PyTorch Engineer specializing in NLP and recurrent neural networks.
We are developing a character/word-level Language Model strictly using LSTM architectures. 
Current status: The model is underfitting, with CrossEntropyLoss plateauing at exactly 0.766.

# Strict Constraints
1. **NO TRANSFORMERS**: The use of Multi-Head Attention, Transformer layers, or any self-attention mechanism is strictly prohibited.
2. **Framework**: Native PyTorch only.
3. **Weight Tying**: The model MUST implement Weight Tying between the `nn.Embedding` layer and the final `nn.Linear` projection layer.
4. **Tokenization**: We are using `jieba` for word-level tokenization. Out-of-vocabulary terms are mapped to `<UNK>`.

# Current Issues to Debug
The training loss is stuck at a plateau (0.766). I need you to analyze and refactor the architecture and hyperparameter initialization based on the following potential causes:
- **Information Bottleneck**: The tied embedding/hidden dimension (`embed_size=64`) might be too rank-deficient to map a large word-level vocabulary.
- **Gradient Flow / Optimizer State**: The learning rate (`5e-5`) might be too low for training an LSTM from scratch with Adam.
- **Premature Regularization**: `dropout=0.2` might be restricting model capacity before it even fits the training manifold.
- **BPTT (Backpropagation Through Time) Handling**: Ensure that hidden states are properly detached or initialized across batches to prevent exploding computational graphs or silent memory leaks.

# Actionable Tasks
1. Refactor `model.py` to ensure the LSTM architecture efficiently handles Weight Tying without shape mismatches.
2. Propose an updated hyperparameter configuration (LR, Hidden Size, Batch Size) optimized for training a word-level LSTM from scratch.
3. Implement a dynamic learning rate scheduler (e.g., `ReduceLROnPlateau` or `CosineAnnealingLR`) in the training loop.
4. Provide code to implement Gradient Clipping (`torch.nn.utils.clip_grad_norm_`) to stabilize the LSTM training.

Maintain professional, clean, and modular code. All explanatory comments in the code MUST be in Traditional Chinese (繁體中文).