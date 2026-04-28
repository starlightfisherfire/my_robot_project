"""
Smoke test for Paper 1 StateNormalizer.

This script verifies:

dummy structured state
→ fit
→ transform
→ inverse_transform
→ save/load

It does not use real MuJoCo or real robot data.
"""

from pathlib import Path

import numpy as np

from src.data.state_normalizer import StateNormalizer


def make_dummy_states(
    batch_size: int = 32,
    history_len: int = 6,
    num_tokens: int = 6,
    raw_token_dim: int = 16,
) -> np.ndarray:
    rng = np.random.default_rng(42)

    states = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(batch_size, history_len, num_tokens, raw_token_dim),
    ).astype(np.float32)

    # sin/cos fields should be in [-1, 1]
    theta = rng.uniform(-np.pi, np.pi, size=(batch_size, history_len, num_tokens))
    states[..., 2] = np.sin(theta)
    states[..., 3] = np.cos(theta)

    # shape one-hot: default shape_T
    states[..., 9] = 1.0
    states[..., 10] = 0.0
    states[..., 11] = 0.0

    # contact flag: binary
    states[..., 14] = rng.integers(
        0, 2, size=(batch_size, history_len, num_tokens)
    ).astype(np.float32)

    # valid flag: all valid for this dummy test
    states[..., 15] = 1.0

    return states


def main():
    states = make_dummy_states()

    normalizer = StateNormalizer()
    normalizer.fit(states)

    normalized = normalizer.transform(states)
    recovered = normalizer.inverse_transform(normalized)

    assert states.shape == normalized.shape
    assert states.shape == recovered.shape

    assert np.isfinite(normalized).all()
    assert np.isfinite(recovered).all()

    # Continuous fields should recover approximately.
    continuous_indices = normalizer.continuous_indices
    max_abs_error = np.max(
        np.abs(states[..., continuous_indices] - recovered[..., continuous_indices])
    )
    assert max_abs_error < 1e-4, max_abs_error

    # Passthrough fields should remain unchanged after transform.
    passthrough_indices = [2, 3, 9, 10, 11, 14, 15]
    passthrough_error = np.max(
        np.abs(states[..., passthrough_indices] - normalized[..., passthrough_indices])
    )
    assert passthrough_error < 1e-6, passthrough_error

    save_path = Path("runs/debug/state_normalizer_v0.json")
    normalizer.save(save_path)

    loaded = StateNormalizer.load(save_path)
    normalized_2 = loaded.transform(states)

    assert np.allclose(normalized, normalized_2)

    print("raw shape:", states.shape)
    print("normalized shape:", normalized.shape)
    print("recovered shape:", recovered.shape)
    print("continuous indices:", continuous_indices)
    print("max inverse error:", float(max_abs_error))
    print("passthrough error:", float(passthrough_error))
    print("save path:", save_path)
    print("state normalizer debug ok")


if __name__ == "__main__":
    main()