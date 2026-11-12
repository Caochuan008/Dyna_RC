# DynaRC

**Dynamics-Robust 4D Radar Relocalization with Rate-Adaptive Sequence Matching**

Chuan Cao, Xingyi Li, Tianchen Deng, and Jingchuan Wang  
School of Automation and Intelligent Sensing, Institute of Medical Robotics  
**Shanghai Jiao Tong University**

DynaRC performs global relocalization in a previously built radar keyframe map.
It combines a static feature context, a Doppler-derived dynamic-evidence mask,
and median ring-key retrieval. Its sequence extension, **Seq-DynaRC**, uses
endpoint-anchored rate alignment (ERA), candidate voting, and ordered descriptor
verification to handle different map/query frame progression rates. The newest
radar scan is registered to the selected keyframe using ICP.

This repository contains the ROS 1 relocalization pipeline, radar preprocessing,
evaluation scripts, and optional baseline adapters. It does not contain the
active-navigation controller or capability-map generation. A prebuilt keyframe
map is required; this package is not a SLAM/map-building system. No GPU, CUDA,
neural-network weights, or model training is required.

The repository directory is `Dyna_RC`; the **catkin package name is `dyna_rc`**.
Use the lowercase name with `roslaunch`, `rosrun`, and `rospack`.

## 1. Environment and dependencies

Target environment: **Ubuntu 20.04 LTS (Focal), ROS 1 Noetic, Python 3**.
The C++ code is built with **C++17** (including `std::filesystem`).

| Component | Target/reference version | Purpose |
|---|---|---|
| Ubuntu | 20.04 LTS, amd64 | Deployment platform |
| ROS | ROS 1 Noetic | Topics, parameters, launch, catkin |
| GCC/G++ | 9.x, C++17 | C++ compilation |
| CMake | 3.16.x on Focal; project minimum 3.10 | Build configuration |
| catkin | Noetic 0.8.x | ROS build and Python installation |
| PCL | 1.10.x | PCD loading, filtering, ICP, preprocessing |
| Eigen | 3.3.7 | Matrix operations and poses |
| Boost | Focal 1.71.x, through PCL/ROS | Transitive C++ dependency |
| OpenCV | 4.2.x | Optional baseline adapters (shared input header); not needed by the core matcher |
| Python | 3.8.x | Recording and offline evaluation |
| NumPy | Focal 1.17.4 series | Numerical evaluation |
| SciPy | Focal 1.3.3 series | Nearest-neighbor queries in histogram analysis |
| Matplotlib | Focal 3.1.2 series | Offline plots; headless `Agg` backend |
| nanoflann | 1.3.2, bundled | Descriptor kd-tree; no separate install |

These are the **target distribution's reference versions**, not an exported
lockfile from every experiment machine. Package revisions vary with apt updates.
This release copy has been prepared on Windows; a fresh Noetic C++ build and bag
replay must be checked on Ubuntu. Use the included environment report to record
the exact versions actually installed on that machine:

```bash
bash scripts/print_environment.sh > environment.txt
```

Version references: [ROS target platforms (REP-3)](https://github.com/ros-infrastructure/rep/blob/master/rep-0003.rst#noetic-ninjemys-may-2020-may-2025),
[Eigen in Focal](https://launchpad.net/ubuntu/focal/+source/eigen3),
[NumPy in Focal](https://launchpad.net/ubuntu/focal/+source/numpy),
[SciPy in Focal](https://launchpad.net/ubuntu/focal/+source/scipy),
and [Matplotlib in Focal](https://launchpad.net/ubuntu/focal/+source/matplotlib).

### Install dependencies

Start with a working ROS Noetic installation on Ubuntu 20.04, then install:

```bash
sudo apt update
sudo apt install build-essential cmake git \
  libpcl-dev libeigen3-dev libopencv-dev \
  ros-noetic-roscpp ros-noetic-rospy ros-noetic-roslaunch ros-noetic-rosbag \
  ros-noetic-sensor-msgs ros-noetic-geometry-msgs ros-noetic-nav-msgs \
  ros-noetic-std-msgs ros-noetic-tf \
  ros-noetic-pcl-ros ros-noetic-pcl-conversions ros-noetic-rostest \
  python3-numpy python3-scipy python3-matplotlib
```

`libopencv-dev` is only used when building the optional baselines. The manifest
lists it so `rosdep` can also prepare that configuration. The core has no
dependency on RaVI-SLAM, `move_base`, Livox drivers, or custom ROS message packages.
The preprocessing nodes use standard `sensor_msgs` messages with radar-specific
fields described below.

## 2. Build

After this repository has been uploaded to `Caochuan008/Dyna_RC`:

```bash
source /opt/ros/noetic/setup.bash
mkdir -p ~/dynarc_ws/src
cd ~/dynarc_ws/src
git clone https://github.com/Caochuan008/Dyna_RC.git
cd ~/dynarc_ws
catkin_make -DCMAKE_BUILD_TYPE=Release -j2
source devel/setup.bash
rospack find dyna_rc
```

For a local copy, put this entire directory under `~/dynarc_ws/src/Dyna_RC`
instead of running `git clone`. Do not nest another catkin workspace inside it.
`-j2` keeps peak memory usage lower when compiling custom PCL point types.

Build the optional RaPlace and ReFeree adapters with:

```bash
catkin_make -DCMAKE_BUILD_TYPE=Release -DDYNA_RC_BUILD_BASELINES=ON -j2
```

The default is `OFF`. These are **4D point-cloud adaptations**, with the original
baseline score scales retained. See [third-party notices](THIRD_PARTY_NOTICES.md).

## 3. Datasets and map format

### SJTU dynamic indoor and outdoor datasets

**Download:** [SJTU datasets, keyframe maps and reference trajectories (Quark Drive)](https://pan.quark.cn/s/fd9142ac371f)  
**Extraction code:** `vTf9`

| Dataset | Environment |
|---|---|
| SJTU-IMR2 | Indoor corridors with pedestrian activity and occlusion |
| SJTU-ROAD / DYNA | Outdoor campus roads under different traffic conditions |

The shared download contains both SJTU datasets, including:

- ROS bags for relocalization.
- Keyframe maps built with **RaVISLAM**: keyframe point clouds (`keyframes/`)
  and the corresponding timestamps and poses.
- Ground-truth/reference trajectories for the relocalization bags.

The data are hosted separately from this source repository. See
[docs/datasets.md](docs/datasets.md) for input and evaluation conventions.

**NTU:** [NTU4DRadLM official dataset](https://github.com/junzhang2016/NTU4DRadLM).

### Map and trajectory format

Example layout below uses illustrative filenames. Set `poses_file`, `pcd_dir`
and `gt_file` to the actual paths in the download; renaming the files is optional.

```text
dataset/
├── poses.txt          # map keyframe poses, in map coordinates
├── keyframes/
│   ├── 0.pcd          # static cloud, in keyframe-local coordinates
│   ├── 0_dyna.pcd     # dynamic cloud, same frame; optional for map loading
│   ├── 1.pcd
│   ├── 1_dyna.pcd
│   └── ...
├── gt.txt             # query reference trajectory, already in the map frame
└── query.bag
```

Both pose text files use eight whitespace-separated columns:

```text
timestamp_seconds x y z qx qy qz qw
```

Translation is in metres; quaternions use **x, y, z, w** order. The map loader
assigns IDs `0, 1, ...` in pose-row order. The first column is a timestamp, not
the PCD filename. Keep the rows chronological and PCD indices consistent. Missing
or filtered-empty static map clouds are skipped, but original keyframe IDs are
retained for reporting. A missing map `_dyna.pcd` is treated as an empty dynamic
cloud. PCD files must provide at least `x`, `y`, `z`, and `intensity`; when using
RCS, the selected return feature must already be carried in the expected field.

Cross-session reference trajectories must be aligned **before** evaluation.
This package does not estimate the LiDAR/radar extrinsics or align two sessions.
Map/query preprocessing, sensor axes, feature channel, and intensity convention
must agree. A bag containing only raw radar points is not sufficient to build the
required keyframe map with this package.

## 4. Run relocalization

### Bags with preprocessed radar streams

For bags containing `/radar_points_inlier` and `/radar_points_outlier`, load
the matching RaVISLAM keyframe map and leave preprocessing disabled.

**Single frame**, terminal 1:

```bash
source ~/dynarc_ws/devel/setup.bash
roslaunch dyna_rc place_recognition.launch \
  data_dir:=/absolute/path/to/dataset \
  run_radar_preprocessing:=false \
  sc_max_radius:=50.0 sc_channel:=2
```

Or use **sequence matching** instead:

```bash
roslaunch dyna_rc seq_place_recognition.launch \
  data_dir:=/absolute/path/to/dataset \
  run_radar_preprocessing:=false \
  sc_max_radius:=50.0 sc_channel:=2
```

Terminal 2, after the map has loaded:

```bash
source ~/dynarc_ws/devel/setup.bash
rosbag play --clock /absolute/path/to/dataset/query.bag
```

Use one matcher at a time for comparative runs, and restart the node and replay
the same query interval for each method. Main launches use simulation time by
default. For live input, set `use_sim_time:=false`.

The example uses the IMR2 descriptor radius of 50 m and the intensity channel
(`sc_channel=2`). For ROAD/DYNA, use `sc_max_radius:=250.0`. Select the feature
channel and map paths for the intended experiment.

### SJTU / ZF bags requiring preprocessing

For a bag containing raw ZF `PointCloud2` measurements, enable
`run_radar_preprocessing:=true`, explicitly set
`radar_preprocessing_launch:=radar_preprocessing.launch`, and set
`radar_topic` to the raw topic in the bag. The default
`/radar_v_points` is only an example. This launch preserves the supplied ZF
extrinsic, ground-filter, RANSAC, and unstable-filter settings; inspect
[launch/radar_preprocessing.launch](launch/radar_preprocessing.launch) for another
platform. Its `use_rcs=true` places the original RCS into output `intensity`.

### Input stream requirements

Leave `run_radar_preprocessing:=false` (the default) and publish both:

- `/radar_points_inlier`: static `PointCloud2` with XYZI fields.
- `/radar_points_outlier`: dynamic `PointCloud2`, including empty messages when no dynamic points exist.

The two messages must share a radar frame and have timestamps within 20 ms.
For this source version, **the single-frame node requires both streams even with
`use_dynamic_sc:=false`**, when it merges them for the ordinary SC baseline.
The sequence node with `use_dynamic_sc:=false` instead consumes `radar_topic`.
Do not run preprocessing twice or replay preprocessed outputs together with a
second publisher on the same topics.

## 5. Parameters and output interfaces

Common launch arguments:

| Argument | Default | Meaning |
|---|---|---|
| `data_dir` | required | Dataset/map root |
| `poses_file`, `pcd_dir` | `data_dir/poses.txt`, `data_dir/keyframes` | Override existing map filenames |
| `use_dynamic_sc` | `true` | DynaRC descriptor/mask path; `false` selects ordinary SC |
| `sc_channel` | `1` | `1`: height; `2`: intensity/RCS; same channel for map and query |
| `sc_max_radius` | `80.0` m | Descriptor radius; set for the dataset |
| `sc_dist_thres` | `1.0` | Single-frame score gate; lower is better |
| `seq_distance_threshold` | `1.0` | Sequence score gate; temporal rejection still applies |
| `seq_type` | `2` | ERA voting matcher; `1` retains the legacy sequence branch |
| `skip_fine_matching` | `false` | Sequence fine-verification ablation |
| `z_min`, `z_max` | `-2.0`, `50.0` m | Map cloud height filter |
| `max_corr_dist`, `max_iter` | `2.0` m, `50` | Final ICP |
| `record` | `false` | Enable CSV recorder with aligned GT |
| `output_csv` | `~/.ros/dynarc/<node>.csv` | Recorder output, with `.meta.json` sidecar |
| `overwrite` | `false` | Explicitly opt in before replacing an existing run |
| `runtime_timing_enabled` | `false` | Save wall-clock stage timings |
| `runtime_output_dir` | `~/.ros/dynarc/runtime` | Per-run runtime directories |
| `metrics_log_file` | empty | Optional legacy text log; parent directory must exist |

`1.0` gates are intended for broad evaluation sweeps, not a calibrated deployment
confidence threshold. The sequence publishes only after sufficient query history,
accepted matching, and converged registration. Thresholds cannot restore a pose
that was never published.

| Topic | Type | Contents |
|---|---|---|
| `/localization_odom` | `nav_msgs/Odometry` | Single-frame pose in `map` |
| `/sequence/localization_odom` | `nav_msgs/Odometry` | Sequence pose for the newest observation |
| `/loop_closure_index` | `std_msgs/Int32` | Original map keyframe ID |
| `/sequence/loop_closure_index` | `std_msgs/Int32` | Original selected endpoint ID |

For both odometry topics, `pose.covariance[0]` carries the matching **distance**
and `pose.covariance[1]` carries the keyframe ID as a numeric value. These two
entries are metadata, **not statistical pose covariance**. Convert the second
entry to an integer when looking up a keyframe. Timestamps refer to the current
query, not the historical map frame. Other diagnostic publishers are retained;
the main evaluation interfaces above are unchanged from the source package.

The source snapshot fixes a 30-frame fine window, invocation stride 5, top-5
per-frame candidates, top-5 hypotheses, and index tolerance 3. These settings
are in `src/seq_place_recognition.cpp`; they are not exposed as launch arguments.
See [implementation notes](docs/implementation_notes.md) for retained search
limits and other snapshot-specific details.

## 6. Repository structure

```text
Dyna_RC/
├── CMakeLists.txt, package.xml
├── src/                    # matchers, preprocessing, GT and offline precompute
├── include/dyna_rc/        # package headers
├── include/scancontext/    # descriptor and bundled kd-tree
├── launch/                 # portable ROS entry points
├── config/                 # sensor conversion/preprocessing parameters
├── scripts/                # CSV recording, plots, environment reporting
├── tests/                  # retained C++/Python/ROS checks
├── third_party/            # optional RaPlace/ReFeree components and notices
└── docs/                   # datasets, evaluation and implementation details
```

## 7. Publication and contact

GitHub: [Caochuan008](https://github.com/Caochuan008).  
Paper: **[TODO: paper/preprint URL and final citation]**.  
Dataset downloads and the extraction code are in Section 3 and
[docs/datasets.md](docs/datasets.md).

The project license and maintainer email still need author confirmation. They
remain explicit placeholders, rather than an assumed license. Review
[LICENSE_NOTICE.md](LICENSE_NOTICE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before uploading the vendored baseline code. This local preparation does not
create a GitHub repository or upload any files.
