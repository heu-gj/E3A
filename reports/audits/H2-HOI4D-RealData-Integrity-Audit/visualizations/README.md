# H2 local-only image diagnostics

All images are withheld from Git. Source RGB, instance overlays, normalized depth,
unknown-color indicators and raw MANO2D keypoints are kept under local_only/.
Metadata, hashes and sizes appear in manifest.json. The source-data snapshots and
resized reference arrays also remain local-only under evidence/local_only/.

Mandatory source frame positions0,75,150,220,224,299 are included. Further positions
are selected automatically by floor/ceil of actual action start/end times times
measured FPS. Detailed selection is in evidence/audit_execution.json.

Depth colors use display normalization only; units are UNKNOWN. Raw2D hand joint
semantics are unverified. Unknown colors remain UNKNOWN; the display indicator is
dilated for visibility, not a change in masks/validity. No objpose projection is
drawn as physical GT. All overlays are diagnostics, not proof of registration,
contact, coordinate conventions or persistent identity.
