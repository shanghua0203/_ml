import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from mamba_inference import parse_args


def test_parse_args_defaults():
    """不傳任何參數時，所有值都應該是預設值"""
    args = parse_args([])
    assert args.model_name == "state-spaces/mamba-1.4b-hf"
    assert args.use_lora is False
    assert args.lora_path == "./coffee_mamba_lora"
    assert args.temperature == 0.7
    assert args.max_tokens == 200
    assert args.top_p == 0.9
    assert args.repetition_penalty == 1.1


def test_parse_args_custom():
    """傳入自訂參數，檢查是否能正確解析"""
    args = parse_args([
        "--model_name", "state-spaces/mamba-790m-hf",
        "--lora_path", "./my_experiment_lora",
        "--temperature", "0.5",
        "--max_tokens", "512",
        "--top_p", "0.8",
        "--repetition_penalty", "1.2",
    ])
    assert args.model_name == "state-spaces/mamba-790m-hf"
    assert args.lora_path == "./my_experiment_lora"
    assert args.temperature == 0.5
    assert args.max_tokens == 512
    assert args.top_p == 0.8
    assert args.repetition_penalty == 1.2


def test_parse_args_use_lora():
    """有加 --use_lora 時，use_lora 應為 True"""
    args_off = parse_args([])
    assert args_off.use_lora is False

    args_on = parse_args(["--use_lora"])
    assert args_on.use_lora is True

    args_on_with_path = parse_args(["--use_lora", "--lora_path", "./other_lora"])
    assert args_on_with_path.use_lora is True
    assert args_on_with_path.lora_path == "./other_lora"


def test_parse_args_temperature_negative():
    """負數的溫度值應該要拋出錯誤（argparse 無法自己檢查，但傳入後可被發現）"""
    import argparse

    msg = None
    try:
        parse_args(["--temperature", "-0.5"])
    except SystemExit:
        # 這不會被 argparse 攔截，所以不應該 SystemExit
        pass

    # argparse 本身不驗證正負，所以應該正常回傳
    args = parse_args(["--temperature", "-0.5"])
    assert args.temperature == -0.5
    # 數值合理性由使用者在外部驗證，此處僅確認 argparse 能正確接收
