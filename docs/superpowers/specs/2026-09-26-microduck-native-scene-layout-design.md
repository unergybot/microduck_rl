# MicroDuck Native MuJoCo Scene Layout Design

## Purpose and decision

Replace the calibrated MicroDuck navigation scene's simplified native MuJoCo
layout with a scene whose desk and doorway have physical collision geometry.
The installed navigation map must represent the same ground footprints. Keep
the door opening traversable and preserve the 17.5 cm visual AprilTag planes.
The user selected physical collision and a new scene revision rather than
changing the meaning of the existing calibrated revision.

This work concerns the native simulator model and its installed navigation
scene. The Web viewer continues to render geometry exported from that model;
the merged Web framing and tag-scale fix in unergy-platform PR #913 is the
baseline, not a new Web feature in this change.

## Current behavior and boundaries

`navigation/environment.py` adds MuJoCo geometry from the installed `Scene`.
For `microduck-navigation-calibration-v1`, obstacle zero is the desk footprint
at x=0.4..0.6 m, y=-0.1..0.3 m. Its tabletop and legs are drawn, while the
door at the `(0, 1, pi/2)` landmark is a visual frame. The runtime switches
`rom_navigation_obstacle_*` collision on for navigation tasks and off for
other actions. Door geoms never collide. The current visual-localization
candidate adds six AprilTag planes separately; each plane has 0.0875 m half
extents. Its tag ID, world pose, texture, and camera calibration are part of
the localization contract.

The existing v1 scene, installed map, historical qualification, and release
evidence remain valid. A simulator using the new source must still accept a v1
installation and render/execute it with the existing v1 geometry and behavior.

## Versioned scene contract

Add `microduck-navigation-calibration-v2`. A versioned canonical Scene fixture
is the source of truth for planar positions and collision footprints. It
contains three ordered obstacles: the existing desk footprint
`[0.4, 0.6] x [-0.1, 0.3]` m, then door posts with footprints
`[-0.325, -0.275] x [0.975, 1.025]` m and
`[0.275, 0.325] x [0.975, 1.025]` m. The v2 validator requires exact
canonical scene content and rejects a missing, moved, or reordered obstacle.
The map grid consumes these same obstacle rectangles. MuJoCo tabletop/legs
and door posts are generated from them, with a header spanning the posts above
the robot's normal standing height. No second independently maintained list
of obstacle coordinates is allowed.

The published v2 fixture keeps the existing `home`, `desk`, and `door`
landmark IDs and their approved positions. The desk landmark remains an
approach pose outside the desk footprint. The door landmark remains centered
in the opening. The doorway has 0.55 m physical clear width. With the current
0.15 m robot radius, 0.05 m clearance, and 0.05 m grid's half-cell diagonal
inflation, its center grid cell remains free. Static room
decoration may be added only when it neither introduces an unmapped collider
nor occludes required visual tags from the qualified approach poses.

AprilTag IDs, physical plane size, surveyed world poses, and their detection
contract remain unchanged for v2. Their planes remain non-colliding. The
calibrated-scene check for the AprilTag probe retains the existing v1 guard
and accepts only the exact canonical v2 scene. The new revision string alone
does not authorize an arbitrary v2 scene.

## Collision and display behavior

During an approved navigation task, the desk and both door posts have MuJoCo
collision enabled. The header is physical geometry above the traversable
opening; its vertical clearance is checked against the MicroDuck model. The
desk and door remain visible in the model exported to the Web viewer. During
non-navigation actions, environmental collision remains disabled to preserve
the qualified action-reset behavior, while the v2 scenery stays visible.
AprilTags and landmark pads are always visual only.

The runtime's fixed-collider guard must recognize only the validated v2
scene-generated collision set. It must continue to reject any other unmapped
static collider. Collision activation may not be inferred from a name prefix
alone: every activated geom must correspond to a validated mapped obstacle or
the explicitly checked overhead header. A v1 installation follows its current
activation and display behavior.

## Verification and acceptance

1. Unit tests compile both v1 and v2 MJCF, assert exact geom identities,
   dimensions, physical collision flags by task mode, 17.5 cm tag planes,
   and v1 compatibility. A v2 scene with a moved or omitted post fails
   validation rather than silently drawing a different map.
2. A map/model consistency test checks all desk and door-post ground
   footprints against the navigation grid, confirms the door opening and the
   desk approach pose are traversable with the approved robot radius and
   clearance, and checks no unmapped static collider is accepted.
3. Rendered camera tests confirm that the unchanged AprilTag IDs remain
   detectable at home, desk approach, and door approach, including the poses
   used by the existing visual-odometry qualification.
4. Re-run the deterministic navigation qualification on the new exact scene
   and simulator source, including door turning, desk detour, blocked-path,
   stop/zero, and collision checks. Re-run the visual-odometry qualification
   before claiming that pose source remains qualified. Report ground-truth
   and visual-odometry results separately; any failing gate blocks release.
5. Stage a new immutable simulator image and v2 scene installation only
   after qualification. Verify the installed scene digest, simulator source
   digest, model/bundle identity, live MuJoCo render, route execution, and
   Boot/simulator health on duale5. Preserve the prior image, v1 installation,
   and rollback evidence. A browser screenshot is visual evidence, not proof
   of physical collision or exact dimensions.

## Scope limits

No robot, policy, observation, or actuator change. No automatic promotion of
the experimental visual pose source. No rewrite of the existing v1 scene
revision, deployed bundle, historical map, or qualification reports. No new
TAIROS landmark IDs. Web framing, camera controls, and PR #913 remain as
deployed unless native-model acceptance reveals a specific Web defect.
