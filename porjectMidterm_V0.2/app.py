"""
app.py —— FastAPI 網頁伺服器（v1.2）
提供 LSTM 語言模型的 Web API 與前端介面
"""

import os
import json
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from model import MyLanguageModel
from text_processor import tokenize


CHECKPOINT_DIR = "checkpointHistory"
VOCAB_DIR = "checkpointHistory"
MAX_CACHED_MODELS = 3


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpointHistory")
VOCAB_DIR = os.path.join(os.path.dirname(__file__), "checkpointHistory")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="LSTM 語言模型 Web 介面")

model_cache = OrderedDict()


class GenerateRequest(BaseModel):
    start_word: str
    model_filename: str
    temperature: float = 0.8
    top_k: int = 15
    max_length: int = 15


class GenerateResponse(BaseModel):
    generated_text: str
    model_filename: str
    temperature: float
    top_k: int
    max_length: int
    inference_time_ms: float


class ModelInfo(BaseModel):
    filename: str
    size_bytes: int
    modified_time: str
    vocab_size: int
    embed_size: int
    hidden_size: int
    num_layers: int


def infer_model_arch(state_dict):
    vocab_size = state_dict["embedding.weight"].shape[0]
    embed_size = state_dict["embedding.weight"].shape[1]
    hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]
    num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))
    return vocab_size, embed_size, hidden_size, num_layers


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    word_to_id = data["word_to_id"]
    id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
    return word_to_id, id_to_word


def load_model_from_checkpoint(checkpoint_path):
    device = torch.device("cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    vocab_size, embed_size, hidden_size, num_layers = infer_model_arch(state)
    model = MyLanguageModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        tie_weights=True,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_cached_model(model_filename):
    if model_filename in model_cache:
        model_cache.move_to_end(model_filename)
        return model_cache[model_filename]

    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到檢查點檔案: {model_filename}")

    model = load_model_from_checkpoint(checkpoint_path)

    vocab_filename = model_filename.replace(".pt", ".json")
    vocab_path = os.path.join(VOCAB_DIR, vocab_filename)
    if os.path.exists(vocab_path):
        word_to_id, id_to_word = load_vocab(vocab_path)
    else:
        root_vocab = os.path.join(os.path.dirname(__file__), "vocab.json")
        if os.path.exists(root_vocab):
            word_to_id, id_to_word = load_vocab(root_vocab)
        else:
            raise FileNotFoundError(f"找不到詞彙表: {vocab_filename} 或 vocab.json")

    while len(model_cache) >= MAX_CACHED_MODELS:
        model_cache.popitem(last=False)

    model_cache[model_filename] = (model, word_to_id, id_to_word)
    return model, word_to_id, id_to_word


def generate_text(model, start_word, word_to_id, id_to_word,
                  max_length=15, temperature=0.8, top_k=15):
    model.eval()
    start_id = word_to_id.get(start_word, 0)
    generated_words = [id_to_word[start_id]]
    current_input = torch.tensor([[start_id]], device=next(model.parameters()).device)
    current_hidden = None
    with torch.no_grad():
        for _ in range(max_length):
            output, current_hidden = model(current_input, current_hidden)
            logits = output.squeeze(0).squeeze(0)
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(0, indices, values)
                logits = mask
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            next_word = id_to_word[next_id]
            generated_words.append(next_word)
            current_input = torch.tensor([[next_id]], device=current_input.device)
    return "".join(generated_words)


@app.get("/api/models")
def list_models():
    if not os.path.isdir(CHECKPOINT_DIR):
        return {"models": []}
    models = []
    for fname in sorted(os.listdir(CHECKPOINT_DIR)):
        if not fname.endswith(".pt"):
            continue
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        stat = os.stat(fpath)
        try:
            state = torch.load(fpath, map_location="cpu", weights_only=True)
            vs, es, hs, nl = infer_model_arch(state)
            del state
        except Exception:
            vs = es = hs = nl = 0
        models.append(ModelInfo(
            filename=fname,
            size_bytes=stat.st_size,
            modified_time=str(stat.st_mtime),
            vocab_size=vs,
            embed_size=es,
            hidden_size=hs,
            num_layers=nl,
        ))
    return {"models": models}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.model_filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="模型檔案必須是 .pt 格式")
    try:
        model, word_to_id, id_to_word = get_cached_model(req.model_filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"載入模型失敗: {str(e)}")
    t0 = time.time()
    result = generate_text(
        model, req.start_word, word_to_id, id_to_word,
        max_length=req.max_length,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    elapsed = (time.time() - t0) * 1000
    return GenerateResponse(
        generated_text=result,
        model_filename=req.model_filename,
        temperature=req.temperature,
        top_k=req.top_k,
        max_length=req.max_length,
        inference_time_ms=round(elapsed, 2),
    )


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
