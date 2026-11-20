# Local-only visualizations

Unapproved dataset-derived images and raw annotation snapshots are excluded from
Git. Five temporal positions per selected camera and additional exact mask-frame
samples are saved locally in local_only/. See evidence/video_alignment.json for
the actual read index, video PTS and distinct camera identity.

Nearest-neighbor resized masks are visualization diagnostics, not new GT. Original
full-resolution annotation dimensions and the scale factors are preserved in the
public statistics. Ego and exo cameras never occupy adjacent entries of one model
time axis. Missing EgoPose/Keystep/Camera annotations are not drawn or synthesized.
