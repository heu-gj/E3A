# E3A

Entity-grounded 3D Action Anticipation

E3A is a research codebase for predicting future human-object interactions from observed video using temporally consistent entity representations and 3D geometry.

## Development policy

Reuse upstream implementations wherever their APIs and licenses permit. E3A should contain integration code, task-specific prediction heads, dataset adapters, losses, tests, and only minimal upstream patches. Do not reimplement foundation models, segmentation trackers, or generic training infrastructure when a maintained upstream implementation already provides them.

## Pinned upstreams

The upstream projects are tracked as Git submodules. Revisions are pinned in the Git tree, not floating branches.

| Upstream | E3A role | Revision |
|---|---|---|
| [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) | Primary geometry/video backbone; official camera and dense-depth heads | 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6 |
| [SAM 3](https://github.com/facebookresearch/sam3) | Optional SAM 3.1 video segmentation/tracking teacher and independent baseline | 31cb7e624e4fe95301b1be1d4e809111593d2e45 |
| [VidEoMT](https://github.com/tue-mps/videomt) | Official clip-consistent entity matcher and video mask criterion, reused through a thin E3A adapter | 40a4fa4a048b00f97f36b3eee66ab62a4e1bea23 |

EntitySAM is a method/paper reference only. Its public repository currently has no repository-level LICENSE file, so its source is not copied or vendored.

## Checkout

    git clone --recurse-submodules https://github.com/heu-gj/E3A.git
    cd E3A
    git submodule status

If already cloned without submodules:

    git submodule update --init --recursive

## Install the primary development environment

Use a PyTorch/CUDA build compatible with the target machine first. The base E3A package follows VGGT-Ω's PyTorch 2.6+ requirements.

    python -m pip install -e ".[dev]"
    python -m pip install -e 'third_party/vggt-omega[train]'

SAM 3.1 is optional and has stricter upstream environment guidance (Python 3.12+, PyTorch 2.10+, CUDA 12.8+ in its current README). Install it only in a compatible environment:

    python -m pip install -e third_party/sam3

SAM 3.1 checkpoints require Hugging Face access approval/authentication unless a local checkpoint is supplied. VGGT-Ω checkpoints also have an access request process. Upstream source checkouts do not include pretrained weights.

VidEoMT's matcher/criterion additionally require SciPy and Detectron2. Detectron2 must be built for the target PyTorch/CUDA version; do not install an arbitrary prebuilt wheel.

## Architecture policy

- VGGT-Ω is the primary geometry encoder. Keep its released checkpoint-compatible modules intact wherever possible.
- SAM 3.1 is initially an offline teacher / segmentation-tracking baseline, not a second online backbone in the E3A forward pass. It has its own large vision encoder and should not be run alongside VGGT-Ω by default.
- E3A-owned code should focus on temporal entity representation, 3D human-object relations, and future target/action/time prediction.
- The exact entity/mask integration point is documented in docs/architecture/upstream-integration.md. Do not change token offsets or head inputs without parity tests.
- Dataset-specific label semantics and sampling rules must be audited before finalizing loss functions or action vocabularies.

## Licensing

- VGGT-Ω: FAIR Noncommercial Research License. Use is restricted to noncommercial research uses under its agreement.
- SAM 3: SAM License. Review its terms before using or redistributing SAM Materials.
- VidEoMT: MIT License.
- EntitySAM: no repository-level LICENSE file was found; no source is copied.

No project-wide E3A license is declared yet. Each upstream's terms continue to apply to its own source and derivative materials. Do not commit datasets, model weights, access tokens, or private annotations.

## Current implementation status

The current foundation includes the pinned VGGT-Ω feature path, native E3A query-to-mask decoding, clip-level persistent entity pooling, official camera/depth unprojection, 3D hand-pose and hand-object interaction heads, target/action/time anticipation heads, a joint multi-task loss, and a VidEoMT adapter that exposes the same clip-consistent track-to-slot assignments to downstream losses. E3AModel runs these heads in one forward pass; E3AMultiTaskLoss returns the single loss_objective required by the reused VGGT-Ω trainer. HOI4D and Ego-Exo4D raw annotation readers, audited action vocabularies, full dataset-backed training configuration, approved-checkpoint parity, and target GPU validation remain open. SAM 3.1 is optional teacher/baseline only and is not loaded by the online model. See docs/implementation/roadmap.md.
