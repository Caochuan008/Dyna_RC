# Datasets

## SJTU dynamic indoor and outdoor datasets

**Download:** [SJTU datasets, keyframe maps and reference trajectories (Quark Drive)](https://pan.quark.cn/s/fd9142ac371f)  
**Extraction code:** `vTf9`

The download contains **SJTU-IMR2** (indoor corridors with pedestrian activity)
and **SJTU-ROAD/DYNA** (outdoor campus roads under different traffic conditions).
Each dataset is supplied with relocalization ROS bags, a **RaVISLAM** keyframe
map (`keyframes/` point clouds plus timestamps and poses), and the reference
trajectory for the relocalization data.

## Loading the data

Download and extract the files locally, then set these launch arguments:

| Argument | File or directory |
|---|---|
| `poses_file` | Map keyframe timestamps and poses |
| `pcd_dir` | Corresponding `keyframes/` point-cloud directory |
| `gt_file` | Reference trajectory for the relocalization bag |

Pose files use `timestamp_seconds x y z qx qy qz qw`; timestamps are in seconds
and translations in metres. Preserve the association between pose rows and
numbered keyframe point clouds. The names `poses.txt` and `gt.txt` in the README
are examples, not a requirement on the download's filenames.

Inspect the bag topics before selecting preprocessing. If the static and dynamic
radar streams are already present, use `run_radar_preprocessing:=false`. For raw
ZF data, enable preprocessing and explicitly select `radar_preprocessing.launch`
with the matching raw topic and calibration. See the README for commands.

Evaluation compares the relocalization output against the reference trajectory
at the same timestamp. Both must be in the map coordinate frame. For separate
mapping and query sessions, use the aligned query reference trajectory.

**NTU:** [NTU4DRadLM official dataset](https://github.com/junzhang2016/NTU4DRadLM).
