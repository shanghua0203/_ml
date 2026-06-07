import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_SEQ_LENGTH
from data_utils import load_dataset


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return load_dataset(tokenizer)


def test_dataset_loads_successfully(dataset):
    assert dataset is not None
    assert len(dataset) > 0


def test_dataset_has_required_columns(dataset):
    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names


def test_dataset_input_ids_are_valid(dataset):
    for i, example in enumerate(dataset):
        ids = example["input_ids"]
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(t, int) for t in ids)
        if i >= 9:
            break


def test_dataset_length_matches_input_ids(dataset):
    for i, example in enumerate(dataset):
        assert len(example["input_ids"]) == example["length"]
        if i >= 9:
            break


def test_all_samples_within_max_length(dataset):
    lengths = dataset["length"]
    assert max(lengths) <= MAX_SEQ_LENGTH
