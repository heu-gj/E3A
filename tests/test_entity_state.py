import torch

from e3a.models.entity_state import PersistentEntityPool


def test_persistent_entity_pool_returns_clip_level_slots_and_valid_centers():
    pool = PersistentEntityPool(embed_dim=8)
    tokens = torch.randn(2, 3, 4, 8, requires_grad=True)
    centers = torch.randn(2, 3, 4, 3)
    valid = torch.ones(2, 3, 4, dtype=torch.bool)
    valid[0, :, 2] = False

    output = pool(
        tokens,
        entity_centers_world=centers,
        entity_geometry_valid=valid,
    )

    assert output["persistent_entity_tokens"].shape == (2, 4, 8)
    assert output["entity_temporal_weights"].shape == (2, 3, 4)
    assert output["persistent_entity_centers_world"].shape == (2, 4, 3)
    assert not output["persistent_entity_geometry_valid"][0, 2]
    torch.testing.assert_close(
        output["persistent_entity_centers_world"][0, 2],
        torch.zeros(3),
    )
    output["persistent_entity_tokens"].sum().backward()
    assert tokens.grad is not None


def test_persistent_entity_pool_supports_single_frame_clips():
    pool = PersistentEntityPool(embed_dim=4)
    tokens = torch.randn(1, 1, 2, 4)
    output = pool(tokens)
    torch.testing.assert_close(output["entity_temporal_weights"], torch.ones(1, 1, 2))
