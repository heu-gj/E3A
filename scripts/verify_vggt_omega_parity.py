"""Verify that the patched optional feature path preserves official VGGT-Ω outputs."""

from __future__ import annotations

import argparse

import torch
from vggt_omega.models import VGGTOmega

from e3a.backbones.vggt_omega import VGGTOmegaAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.resolution % 16:
        raise ValueError("resolution must be divisible by VGGT-Ω patch size 16.")

    model = VGGTOmega(autocast=False).to(args.device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    adapter = VGGTOmegaAdapter(model=model, freeze=False).eval()
    images = torch.rand(
        1,
        args.frames,
        3,
        args.resolution,
        args.resolution,
        device=args.device,
    )

    with torch.inference_mode():
        reference = model(images)
        adapted = adapter.forward_with_features(images, extra_tokens=None)

    expected_keys = set(reference)
    actual_keys = set(adapted)
    if actual_keys != expected_keys | {"fused_features"}:
        raise AssertionError(
            "Unexpected optional-feature output contract: "
            f"reference={sorted(expected_keys)}, adapted={sorted(actual_keys)}"
        )
    if "extra_tokens" in adapted:
        raise AssertionError("No extra-token output should be returned when extra_tokens=None.")

    for key, expected in reference.items():
        torch.testing.assert_close(
            adapted[key],
            expected,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda message, key=key: f"VGGT-Ω parity failed for {key}: {message}",
        )

    fused = adapted["fused_features"]
    expected_feature_size = (args.resolution // 4, args.resolution // 4)
    if tuple(fused.shape[-2:]) != expected_feature_size:
        raise AssertionError(
            f"Unexpected fused feature size: {tuple(fused.shape[-2:])}, "
            f"expected {expected_feature_size}"
        )
    if not torch.isfinite(fused).all():
        raise AssertionError("VGGT-Ω fused features contain non-finite values.")

    print("VGGT-Ω no-extra-token output parity: PASS")
    print("Fused feature shape:", tuple(fused.shape))
    print("Device:", args.device)
    print("PyTorch:", torch.__version__)


if __name__ == "__main__":
    main()
