import torch

from e3a.models.token_bank import E3ATokenBank


def test_token_bank_shapes_and_slices():
    bank = E3ATokenBank(embed_dim=32, num_entities=5, num_hands=2)
    timestamps = torch.tensor([[0.0, 0.1, 0.3], [0.0, 0.2, 0.4]])

    tokens = bank(timestamps)
    parts = bank.split(tokens)

    assert tokens.shape == (2, 3, 8, 32)
    assert parts["entity_tokens"].shape == (2, 3, 5, 32)
    assert parts["hand_tokens"].shape == (2, 3, 2, 32)
    assert parts["future_tokens"].shape == (2, 3, 1, 32)


def test_token_bank_uses_time_and_backpropagates():
    bank = E3ATokenBank(embed_dim=16, num_entities=3)
    timestamps = torch.tensor([[0.0, 0.5, 1.0]])
    tokens = bank(timestamps)
    loss = tokens[:, 1:].square().mean()
    loss.backward()

    assert bank.entity_tokens.grad is not None
    assert bank.time_embedding[0].weight.grad is not None


def test_token_bank_rejects_wrong_timestamp_shape():
    bank = E3ATokenBank(embed_dim=16, num_entities=3)
    try:
        bank(torch.tensor([0.0, 0.5]))
    except ValueError as exc:
        assert "shape [B,T]" in str(exc)
    else:
        raise AssertionError("Expected invalid timestamp shape to fail.")


def test_token_bank_rejects_non_finite_timestamps():
    bank = E3ATokenBank(embed_dim=16, num_entities=3)
    try:
        bank(torch.tensor([[0.0, float("nan")]]))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected non-finite timestamps to fail.")
