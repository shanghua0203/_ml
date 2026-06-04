import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from data_utils import load_dataset


def test_dataset_loads_successfully():
    ds = load_dataset()
    assert ds is not None
    assert len(ds) > 0


def test_dataset_has_required_columns():
    ds = load_dataset()
    assert "input_ids" in ds.column_names
    assert "attention_mask" in ds.column_names


def test_dataset_input_ids_are_valid():
    ds = load_dataset()
    for i, example in enumerate(ds):
        ids = example["input_ids"]
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(t, int) for t in ids)
        if i >= 9:
            break


def test_dataset_length_matches_input_ids():
    ds = load_dataset()
    for i, example in enumerate(ds):
        assert len(example["input_ids"]) == example["length"]
        if i >= 9:
            break


def test_all_samples_within_max_length():
    from config import MAX_SEQ_LENGTH
    ds = load_dataset()
    lengths = ds["length"]
    assert max(lengths) <= MAX_SEQ_LENGTH
