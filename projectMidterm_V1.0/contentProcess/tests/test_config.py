import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import config


def test_model_name_is_string():
    assert isinstance(config.MODEL_NAME, str)
    assert len(config.MODEL_NAME) > 0


def test_lora_r_is_positive_int():
    assert isinstance(config.LORA_R, int)
    assert config.LORA_R > 0


def test_lora_alpha_is_positive_int():
    assert isinstance(config.LORA_ALPHA, int)
    assert config.LORA_ALPHA > 0


def test_lora_dropout_in_range():
    assert 0 <= config.LORA_DROPOUT < 1


def test_target_modules_not_empty():
    assert isinstance(config.TARGET_MODULES, list)
    assert len(config.TARGET_MODULES) > 0


def test_batch_size_is_positive():
    assert config.BATCH_SIZE >= 1


def test_gradient_accumulation_is_positive():
    assert config.GRADIENT_ACCUMULATION_STEPS >= 1


def test_learning_rate_is_positive():
    assert config.LEARNING_RATE > 0


def test_num_epochs_is_positive():
    assert config.NUM_EPOCHS > 0


def test_max_seq_length_is_reasonable():
    assert 128 <= config.MAX_SEQ_LENGTH <= 8192
