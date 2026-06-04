import json
import re
import sys
from pathlib import Path

import ollama
from tqdm import tqdm

INPUT_FILE = "data/raw/input.txt"
OUTPUT_FILE = "data/processed/output.jsonl"
MODEL = "gemma3:4b"
CHUNK_MIN = 200
CHUNK_MAX = 500

SYSTEM_PROMPT = (
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


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[。！？\n])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str) -> list[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    raw_chunks = []

    for para in paragraphs:
        if len(para) <= CHUNK_MAX:
            raw_chunks.append(para)
        else:
            lines = [l.strip() for l in para.split('\n') if l.strip()]
            for line in lines:
                if len(line) <= CHUNK_MAX:
                    raw_chunks.append(line)
                else:
                    sentences = split_sentences(line)
                    temp = ""
                    for sent in sentences:
                        if len(temp) + len(sent) > CHUNK_MAX:
                            if temp:
                                raw_chunks.append(temp)
                            temp = sent
                        else:
                            temp += sent
                    if temp:
                        raw_chunks.append(temp)

    merged = []
    current = ""
    for chunk in raw_chunks:
        if not current:
            current = chunk
        elif len(current) + len(chunk) <= CHUNK_MAX:
            current += "\n" + chunk
        else:
            merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    if len(merged) > 1 and len(merged[-1]) < CHUNK_MIN:
        merged[-2] += "\n" + merged.pop()

    return merged


def generate_qa_pairs(chunk: str) -> list[dict]:
    try:
        response = ollama.generate(
            model=MODEL,
            prompt=f"請根據以下文本，生成多組問答對：\n{chunk}",
            system=SYSTEM_PROMPT,
            format="json",
        )
        text = response.get("response", "").strip()
        if not text:
            print("  [WARN] 模型回傳空內容")
            return []

        data = json.loads(text)

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
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON 解析失敗: {e}")
        print(f"  [RAW] {text[:200]}...")
        return []
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []


def main():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"[ERROR] 找不到輸入檔案: {INPUT_FILE}")
        print(f"  請將 input.txt 放在目前目錄下")
        sys.exit(1)

    print(f"[INFO] 讀取 {INPUT_FILE} ...")
    text = input_path.read_text(encoding="utf-8")
    print(f"[INFO] 總字數: {len(text)}")

    print("[INFO] 切割文本 ...")
    chunks = chunk_text(text)
    print(f"[INFO] 總共切割成 {len(chunks)} 個區塊")

    total_pairs = 0
    total_fail = 0
    chunk_ok = 0
    chunk_fail = 0
    output_path = Path(OUTPUT_FILE)

    with output_path.open("a", encoding="utf-8") as f:
        for i, chunk in enumerate(tqdm(chunks, desc="處理進度")):
            pairs = generate_qa_pairs(chunk)
            if pairs:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    f.flush()
                total_pairs += len(pairs)
                chunk_ok += 1
            else:
                chunk_fail += 1
                total_fail += 1

    print(f"\n[完成] 成功區塊: {chunk_ok}/{len(chunks)}")
    print(f"[完成] 總產出 QA 筆數: {total_pairs}")
    print(f"[完成] 失敗區塊: {chunk_fail}")
    print(f"[完成] 輸出檔案: {output_path.resolve()}")


if __name__ == "__main__":
    main()
