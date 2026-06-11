"""
zhqa_generator.core
~~~~~~~~~~~~~~~~~~~
將中文純文字切割並透過 Ollama LLM 自動生成 QA 問答對。

基本用法：
    from zhqa_generator import chunk_text, generate_qa_pairs

    text = open("article.txt", encoding="utf-8").read()
    chunks = chunk_text(text, chunk_min=200, chunk_max=500)
    pairs = generate_qa_pairs(chunks, model="gemma3:4b")
"""

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

# ── 預設系統提示詞 ────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "你是一個專業的問答生成助手。請根據提供的文本，生成多組高品質的問答對。\n\n"
    "要求：\n"
    "1. 仔細閱讀文本內容\n"
    "2. 生成 3 到 5 組問答對，從不同角度涵蓋文本內容\n"
    "3. 問題必須基於文本，措辭具體且明確\n"
    "4. 答案必須詳細、準確、完整\n"
    "5. 每組問答對的內容不能重複\n"
    "6. 請使用繁體中文（正體字）回答，不要使用簡體字\n\n"
    "你必須嚴格按照以下 JSON 格式輸出，不能包含任何其他文字、標記或程式碼區塊：\n"
    '{"qa_pairs": [{"prompt": "問題1", "completion": "答案1"}, {"prompt": "問題2", "completion": "答案2"}]}'
)


# ── 文本切割 ──────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """以中文句號、驚嘆號、問號或換行為邊界分句。"""
    sentences = re.split(r'(?<=[。！？\n])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_min: int = 200,
    chunk_max: int = 500,
) -> list[str]:
    """
    將長文本切割為適合 LLM 輸入的區塊。

    Args:
        text:      原始純文字（任意長度）。
        chunk_min: 區塊最小字元數；最後一個過短的區塊會合併至前一個。
        chunk_max: 區塊最大字元數；超過此長度會繼續切分。

    Returns:
        字串列表，每個元素為一個切割好的文本區塊。
    """
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    raw_chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= chunk_max:
            raw_chunks.append(para)
        else:
            lines = [line.strip() for line in para.split('\n') if line.strip()]
            for line in lines:
                if len(line) <= chunk_max:
                    raw_chunks.append(line)
                else:
                    sentences = split_sentences(line)
                    temp = ""
                    for sent in sentences:
                        if len(temp) + len(sent) > chunk_max:
                            if temp:
                                raw_chunks.append(temp)
                            temp = sent
                        else:
                            temp += sent
                    if temp:
                        raw_chunks.append(temp)

    # 合併過短的相鄰區塊
    merged: list[str] = []
    current = ""
    for chunk in raw_chunks:
        if not current:
            current = chunk
        elif len(current) + len(chunk) <= chunk_max:
            current += "\n" + chunk
        else:
            merged.append(current)
            current = chunk
    if current:
        merged.append(current)

    # 最後一個區塊若過短，合併至前一個
    if len(merged) > 1 and len(merged[-1]) < chunk_min:
        merged[-2] += "\n" + merged.pop()

    return merged


# ── QA 生成 ───────────────────────────────────────────────────────────────────

def generate_qa_for_chunk(
    chunk: str,
    model: str = "gemma3:4b",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict]:
    """
    對單一文本區塊呼叫 Ollama，回傳 QA 對列表。

    Args:
        chunk:         單一文本區塊。
        model:         Ollama 模型名稱（預設 ``gemma3:4b``）。
        system_prompt: 系統提示詞。

    Returns:
        ``[{"prompt": str, "completion": str}, ...]``，解析失敗時回傳空列表。
    """
    try:
        import ollama as _ollama
    except ImportError as exc:
        raise ImportError(
            "需要安裝 ollama：pip install ollama"
        ) from exc

    try:
        response = _ollama.generate(
            model=model,
            prompt=f"請根據以下文本，生成多組問答對：\n{chunk}",
            system=system_prompt,
            format="json",
        )
        raw = response.get("response", "").strip()
        if not raw:
            print("  [WARN] 模型回傳空內容")
            return []

        data = json.loads(raw)
        pairs = data.get("qa_pairs", [])
        if not pairs:
            pairs = [data]

        valid = []
        for pair in pairs:
            prompt = pair.get("prompt")
            completion = pair.get("completion")
            if prompt and completion:
                valid.append({"prompt": prompt, "completion": completion})
            else:
                print(f"  [WARN] 跳過無效 QA: {pair}")

        return valid

    except json.JSONDecodeError as exc:
        print(f"  [WARN] JSON 解析失敗: {exc}")
        return []
    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] {exc}")
        return []


def generate_qa_pairs(
    chunks: list[str],
    model: str = "gemma3:4b",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    show_progress: bool = True,
) -> list[dict]:
    """
    對多個文本區塊批次生成 QA 對。

    Args:
        chunks:        由 :func:`chunk_text` 產生的區塊列表。
        model:         Ollama 模型名稱。
        system_prompt: 系統提示詞（不傳則使用預設中文提示）。
        show_progress: 是否顯示 tqdm 進度條。

    Returns:
        所有成功的 QA 對，格式為 ``[{"prompt": str, "completion": str}, ...]``。
    """
    all_pairs: list[dict] = []
    iterator = tqdm(chunks, desc="生成 QA") if show_progress else chunks

    for chunk in iterator:
        pairs = generate_qa_for_chunk(chunk, model=model, system_prompt=system_prompt)
        all_pairs.extend(pairs)

    return all_pairs


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI 入口：zhqa-generate"""
    parser = argparse.ArgumentParser(
        prog="zhqa-generate",
        description="將中文純文字透過 Ollama 生成 QA 問答對（供 LLM 微調使用）",
    )
    parser.add_argument("--input", "-i", required=True, help="輸入的純文字檔路徑")
    parser.add_argument(
        "--output", "-o", default="output.jsonl", help="輸出的 JSONL 檔路徑（預設：output.jsonl）"
    )
    parser.add_argument(
        "--model", "-m", default="gemma3:4b", help="Ollama 模型名稱（預設：gemma3:4b）"
    )
    parser.add_argument("--chunk-min", type=int, default=200, help="切割區塊最小字元數（預設：200）")
    parser.add_argument("--chunk-max", type=int, default=500, help="切割區塊最大字元數（預設：500）")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 找不到輸入檔案：{args.input}")
        sys.exit(1)

    print(f"[INFO] 讀取 {args.input} ...")
    text = input_path.read_text(encoding="utf-8")
    print(f"[INFO] 總字數：{len(text)}")

    print("[INFO] 切割文本 ...")
    chunks = chunk_text(text, chunk_min=args.chunk_min, chunk_max=args.chunk_max)
    print(f"[INFO] 共切割成 {len(chunks)} 個區塊")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    chunk_ok = 0
    chunk_fail = 0

    with output_path.open("a", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="處理進度"):
            pairs = generate_qa_for_chunk(chunk, model=args.model)
            if pairs:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    f.flush()
                total_pairs += len(pairs)
                chunk_ok += 1
            else:
                chunk_fail += 1

    print(f"\n[完成] 成功區塊：{chunk_ok}/{len(chunks)}")
    print(f"[完成] 總產出 QA 筆數：{total_pairs}")
    print(f"[完成] 失敗區塊：{chunk_fail}")
    print(f"[完成] 輸出檔案：{output_path.resolve()}")


if __name__ == "__main__":
    main()
