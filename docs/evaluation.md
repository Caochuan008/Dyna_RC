# Additional evaluation tools

## Recorder only

When the matcher is already running:

```bash
roslaunch dyna_rc localization_recall_recorder.launch \
  keyframe_poses_file:=/data/map/poses.txt gt_file:=/data/query/gt_in_map.txt \
  localization_topic:=/sequence/localization_odom \
  output_csv:=$HOME/dynarc_results/seq.csv algorithm_name:=Seq-DynaRC
```

The recorder stores original integer timestamp nanoseconds and interpolates the
reference pose. `gt_time_offset` shifts the query timestamp for GT lookup;
`max_gt_gap` rejects interpolation over long reference gaps. Map and GT hashes
are saved for consistency checks. The matched index is retained separately from
the reported localization pose; select `error_source=localization` for output
pose accuracy.

## Histograms

```bash
roslaunch dyna_rc localization_error_histograms.launch \
  keyframe_poses_file:=/data/map/poses.txt gt_file:=/data/query/gt_in_map.txt \
  localization_topic:=/localization_odom min_bin_count:=100
```

This tool groups the query's interpolated reference pose by its offset from the
XY-nearest map keyframe. Output-pose success is a separate comparison to GT.
`success_xy_m` and `success_yaw_deg` default to 3 m and 15 degrees. Plotting excludes
sparsely populated bins and samples meeting both near-zero offset cutoffs.
See the launch arguments for yaw groups, bin counts, and filtering controls.

## Optional non-neighbor table

Run before bag playback; the node exits after writing the table:

```bash
roslaunch dyna_rc non_neighbor_precompute.launch \
  poses_file:=/data/map/poses.txt pcd_dir:=/data/map/keyframes \
  output_csv:=/data/map/non_neighbor.csv \
  use_dynamic_sc:=true sc_max_radius:=80 \
  success_xy_m:=3.0 success_yaw_deg:=15.0
```

A non-neighbor violates the XY **or** yaw neighborhood condition. Its table score
is the smallest valid descriptor distance to any such keyframe, representing
the most confusable non-neighbor. Pass the table using
`non_neighbor_csv:=/data/map/non_neighbor.csv` to the histogram launch. Match map,
range, z filter, descriptor definition, and success thresholds. This source
snapshot's precompute node uses the height channel only; see implementation notes.
Keep `respawn=false` for this one-shot node.

## Native-score comparison

`localization_recall_comparison.py` accepts ROS parameters or command-line
arguments. Native mode sweeps each run's unique observed scores. Shared-grid mode
uses `score_min`, `score_max`, and `score_step`; it is unsuitable for comparing
arbitrarily different distance scales without an explicit plan.

The output names include `comparison_row.pdf`, `recall_vs_acceptance.pdf`, PNG
versions, and curve CSVs. Always report the denominator. With `accepted`, failed
or withheld upstream outputs are absent, and the result measures reliability
of retained published poses. A fixed-denominator option exists, but a union of
recorded timestamps is not necessarily the set of all input scans.

## Runtime

Use `runtime_timing_enabled=true` and retain `metadata.txt`, `frames.csv`, and
`summary.csv` together. The files separate query attempts, publications and
individual ICP calls. State whether rejected attempts are included when quoting
stage means. Query stride and window warm-up mean that input count, attempted
match count and published pose count need not agree.
