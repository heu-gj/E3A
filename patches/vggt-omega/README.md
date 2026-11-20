# VGGT-Ω E3A compatibility patch

Target upstream commit: 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6

Patch: 0001-extra-tokens-and-dense-features.patch

Changes are deliberately opt-in:

1. Aggregator.forward accepts optional per-frame extra prefix tokens. The original camera/register parameters and their prefix length are unchanged.
2. Register-only inter-frame attention receives the expanded prefix length explicitly. No mutable per-forward state is stored on the Aggregator.
3. DenseHead.forward accepts return_features=False. Its default two-output contract is unchanged; when requested, it additionally returns the fused 1/4-resolution feature as [B,T,C,Hf,Wf].

The patch does not alter upstream parameter names or tensor shapes and does not add parameters to the upstream checkpoint. Run the no-extra-token parity test before training.
