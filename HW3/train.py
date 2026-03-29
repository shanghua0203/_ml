"""
訓練腳本：使用 nn0.py 的極小神經網路
使用數值穩定的 cross_entropy 函數
"""

import random
from nn0 import Value, Adam, linear, rmsnorm, cross_entropy


class TinyModel:
    """
    極簡神經網路：1 層、極小維度
    """

    def __init__(self, vocab_size=10, embed_dim=4, n_head=2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = 1
        self.block_size = 8

        self.tok_emb = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]

        self.ln_scale = [Value(1.0) for _ in range(embed_dim)]
        self.ln_bias = [Value(0.0) for _ in range(embed_dim)]

        self.head = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]

    def __call__(self, token_id, pos_id, keys, values):
        x = self.tok_emb[token_id]

        x_normed = rmsnorm(x)
        x_normed = [
            x_normed[i] * self.ln_scale[i] + self.ln_bias[i]
            for i in range(len(x_normed))
        ]

        logits = linear(x_normed, self.head)
        return logits


def generate_tokens(vocab_size=10, length=15):
    """生成隨機整數序列作為 tokens"""
    return [random.randint(0, vocab_size - 1) for _ in range(length)]


def train_step(model, optimizer, tokens):
    """
    一步訓練：forward → loss (stable) → backward → Adam update
    """
    n = min(model.block_size, len(tokens) - 1)
    keys = [[] for _ in range(model.n_layer)]
    values = [[] for _ in range(model.n_layer)]

    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = model(token_id, pos_id, keys, values)
        loss_t = cross_entropy(logits, target_id)
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = optimizer.lr
    optimizer.step(lr_override=lr_t)

    return loss.data


def main():
    random.seed(42)

    vocab_size = 10
    embed_dim = 4

    model = TinyModel(vocab_size=vocab_size, embed_dim=embed_dim)

    params = []
    for emb in model.tok_emb:
        params.extend(emb)
    for p in model.ln_scale:
        params.append(p)
    for p in model.ln_bias:
        params.append(p)
    for row in model.head:
        params.extend(row)

    optimizer = Adam(params, lr=0.1)

    tokens = generate_tokens(vocab_size=vocab_size, length=15)
    print(f"Tokens: {tokens}")
    print("-" * 40)

    num_steps = 100
    for step in range(num_steps):
        loss = train_step(model, optimizer, tokens)
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step:3d} | Loss: {loss:.6f}")

    print("-" * 40)
    print("Training completed!")


if __name__ == "__main__":
    main()
