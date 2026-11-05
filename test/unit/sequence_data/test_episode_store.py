"""Episode boundaries, provenance and integrity are dynamics-data invariants."""

import numpy as np
import pytest
from genedynamics.learning.datasets import Episode, EpisodeStore, SequenceSampler


def episode(identifier="walk", split="train"):
    arrays = {
        "state": np.arange(3, dtype=np.float32)[:, None],
        "next_state": np.arange(1, 4, dtype=np.float32)[:, None],
        "action": np.ones((3, 1), np.float32),
        "terminated": np.array([False, False, True]),
        "truncated": np.zeros(3, bool),
    }
    return Episode(
        identifier, "original_capture", split, arrays, {"controller": "fixed"}
    )


def test_roundtrip_masks_and_original_sequence_split(tmp_path):
    store = EpisodeStore(tmp_path)
    store.append(episode())
    restored = store.load(store.records()[0])
    np.testing.assert_array_equal(restored.arrays["next_state"], [[1], [2], [3]])
    sample = SequenceSampler([restored]).sample(8, 4, burn_in=1)
    assert not sample.loss_mask[:, 0].any()
    for row, start in enumerate(sample.start_indices):
        np.testing.assert_array_equal(sample.valid[row], start + np.arange(5) < 3)
    with pytest.raises(FileExistsError):
        store.append(episode())
    with pytest.raises(ValueError, match="multiple dataset splits"):
        store.append(episode("held_out", "test"))


def test_checksum_and_terminal_boundary(tmp_path):
    store = EpisodeStore(tmp_path)
    store.append(episode())
    record = store.records()[0]
    (tmp_path / record["file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        store.load(record)
    original = episode()
    invalid = dict(original.arrays, terminated=np.array([True, False, True]))
    with pytest.raises(ValueError, match="episode end"):
        Episode("broken", "source", "train", invalid, {})


def test_concurrent_collectors_cannot_cross_original_sequence_splits(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def append(split):
        try:
            EpisodeStore(tmp_path).append(episode(split, split))
            return "committed"
        except ValueError:
            return "split_rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, ["train", "test"]))
    assert sorted(results) == ["committed", "split_rejected"]
    assert len(EpisodeStore(tmp_path).records()) == 1
