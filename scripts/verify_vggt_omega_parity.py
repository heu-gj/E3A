"""Check E3A's no-extra-token path against the pinned official VGGT-Ω forward."""

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

    keys = ("camera_and_register_tokens", "pose_enc", "depth", "depth_conf")
    for key in keys:
        if key not in reference or key not in adapted:
            raise KeyError(f"Missing required parity output: {key}")
        torch.testing.assert_close(
            adapted[key],
            reference[key],
            rtol=1e-5,
            atol=1e-5,
            msg=lambda message: f"VGGT-Ω parity failed for {key}: {message}",
        )

    expected_feature_size = (args.resolution // 4, args.resolution // 4)
    if tuple(adapted["fused_features"].shape[-2:]) != expected_feature_size:
        raise AssertionError(
            "Unexpected fused feature size: "
            f"{tuple(adapted['fused_features'].shape[-2:])}, "
            f"expected {expected_feature_size}"
        )

    print("VGGT-Ω no-extra-token parity: PASS")
    print("Fused feature shape:", tuple(adapted["fused_features"].shape))


if __name__ == "__main__":
    main()
