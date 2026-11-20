# VGGT-Ω E3A compatibility patches

Target upstream commit: 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6

The patches are applied in numerical order by scripts/apply_patches.sh.

## 0001 — Extra tokens and dense features

1. Aggregator.forward accepts optional per-frame extra prefix tokens.
2. The original camera/register prefix length remains a stable model property. The expanded prefix length is local to the current forward call.
3. Register-only inter-frame attention includes E3A tokens as prefix tokens without mutating module state.
4. DenseHead.forward(return_features=False) preserves the default two-output contract. When enabled, it returns the fused 1/4-resolution feature as [B,T,C,Hf,Wf], including frame chunking.

## 0002 — Official top-level forward feature API

1. VGGTOmega.forward accepts optional extra_tokens and return_features keyword arguments.
2. The default model(images) call preserves the released output keys and behavior.
3. CameraHead and TextAlignmentHead see only the original camera/register prefix.
4. DenseHead receives the expanded prefix so E3A tokens are not interpreted as image patches.
5. The optional extra_tokens output contains the final concatenated frame/inter-frame representation; fused_features is returned only when explicitly requested.

Neither patch adds or changes upstream parameters, parameter names, or checkpoint tensor shapes. The strict upstream checkpoint load remains the source of truth. CI tests compare the default forward against the pinned upstream contract and exercise prefix indexing and feature chunking.
