# E3A

**Entity-grounded 3D Action Anticipation**

E3A is a research codebase for predicting future human-object interactions from observed video, using temporally consistent entity representations and 3D geometry.

> Development policy: reuse upstream implementations wherever their APIs and licenses permit. E3A should contain integration code, task-specific heads/losses/data adapters, tests, and minimal upstream patches—not reimplementations of foundation models.

## Upstream code

The following repositories are tracked as pinned Git submodules (see `.gitmodules`):

| Upstream | Role in E3A | Pinned revision |
|---|---|---|
| [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) | Primary geometry/video backbone and camera/depth heads | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` |
| [SAM 3](https://github.com/facebookresearch/sam3) | Optional video segmentation/tracking teacher and baseline; current upstream includes SAM 3.1 multiplex | `31cb7e624e4fe95301b1be1d4e809111593d2e45` |
| [VidEoMT](https://github.com/tue-mps/videomt) | Optional reference implementation for query propagation and video-mask training | `40a4fa4a048b00f97f36b3eee66ab62a4e1bea23` |

EntitySAM is tracked as a paper/method reference only for now. Its public repository does not expose a repository-level LICENSE file, so E3A will not vendor or copy its source until reuse permissions are confirmed.

## Repository layout

The E3A-owned implementation lives under `src/e3a/`. Upstream repositories remain unmodified in `third_party/`; any required compatibility changes must be kept as small, reviewable patch files under `patches/` and covered by parity tests.

## Checkout

```bash
git clone --recurse-submodules https://github.com/heu-gj/E3A.git
cd E3A
git submodule status
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Model checkpoints

Upstream code and pretrained weights are separate. In particular, VGGT-Ω and SAM 3 checkpoints may require accepting the respective model terms and requesting Hugging Face access. Checkpoints must be downloaded by the user into a local, ignored directory; do not commit model weights or access tokens.

## Licensing

Upstream licenses are not interchangeable:

- VGGT-Ω is distributed under the FAIR Noncommercial Research License. Its use is restricted to noncommercial research uses under that agreement.
- SAM 3 is distributed under the SAM License. Review its terms before using or redistributing SAM Materials.
- VidEoMT is distributed under the MIT License.
- EntitySAM currently has no repository-level LICENSE file visible in its public repository; its code is therefore not copied into E3A.

No project-wide E3A license is declared yet. Third-party terms continue to apply to their respective submodules and derived materials.

## Status

The repository is being initialized. The first implementation milestone is a reproducible upstream checkout and a verified VGGT-Ω adapter with checkpoint-loading and output-parity tests. No claim of a trainable E3A model is made until those tests and the dataset/label contracts pass.
