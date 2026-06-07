#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "========================================="
echo "  ContentProcess 專案整合測試"
echo "========================================="
echo ""

# ── 1. 語法檢查 ──
echo "▶ [1/5] 語法檢查"
python3 -m py_compile scripts/config.py
python3 -m py_compile scripts/model_utils.py
python3 -m py_compile scripts/data_utils.py
python3 -m py_compile scripts/train.py
python3 -m py_compile scripts/inference_test.py
python3 -m py_compile scripts/mamba_inference.py
python3 -m py_compile scripts/main.py
python3 -m py_compile scripts/analyze_training_logs.py
python3 -m py_compile tests/__init__.py
python3 -m py_compile tests/test_config.py
python3 -m py_compile tests/test_data_utils.py
python3 -m py_compile tests/test_model_utils.py
python3 -m py_compile tests/test_mamba_inference.py
python3 -m py_compile tests/test_main.py
echo "  ✅ 語法檢查通過"
echo ""

# ── 2. 單元測試 ──
echo "▶ [2/5] 單元測試"
python3 -m pytest tests/ -v --tb=short
echo ""

# ── 3. 推理測試 ──
echo "▶ [3/5] 系統測試 — Mamba 模型推理測試"
python3 scripts/inference_test.py
echo ""

# ── 4. 資料集完整性檢查 ──
echo "▶ [4/5] 資料集完整性檢查"
python3 -c "
from pathlib import Path
from datasets import load_from_disk

ds_path = 'data/processed/tokenized_dataset'
assert Path(ds_path).exists(), f'❌ 找不到 {ds_path}'
ds = load_from_disk(ds_path)
assert len(ds) > 0, '❌ 資料集為空'
print(f'  ✅ 資料集路徑: {ds_path}')
print(f'  ✅ 筆數: {len(ds)}')
print(f'  ✅ 欄位: {ds.column_names}')
print(f'  ✅ 總 tokens: {sum(ds[\"length\"])}')
"
echo ""

# ── 5. 最終報告 ──
echo "▶ [5/5] 測試結果"
echo "========================================="
echo "  ✅ 全部測試通過"
echo "========================================="
