# Source snapshot and release scope

This package is a release-oriented copy of the supplied `place_recognition`
package. Descriptor construction, channel selection, sequence voting/alignment,
ICP, and preprocessing math have not been redesigned during packaging.

## Packaging changes

- Catkin project/package renamed to `dyna_rc`; public headers moved to `include/dyna_rc`.
- Active manager, capability-map generator, controller forwarding, desktop startup,
  occupancy-image publishing, and their launch/documentation/RViz assets omitted.
- Machine-specific paths removed from runtime defaults and examples. Optional legacy
  text logs use `metrics_log_file`; GT file paths must be explicitly provided.
- Main launches use required dataset paths, optional preprocessing and optional recording.
- All retained executable targets and launch/config resources have install rules.
- Python script shebangs and LF line endings are retained for Ubuntu.
- RaPlace/ReFeree remain optional build targets; raw results, caches and large data are omitted.

Topic names and score/index conventions remain compatible with the original
evaluation pipeline. The global parameters used by the two core matchers are
retained. Run separate method evaluations in separate replays; the launches are
not designed to isolate multiple global parameter sets under one ROS master.

## Source constants worth checking for reproduction

- Descriptor: 20 rings, 20 sectors, 120-degree angular domain; channel `1` is
  shifted height and channel `2` is intensity. The preserved implementation is
  the supplied snapshot, not a reimplementation inferred from the paper.
- Sequence node: fine window 30, invocation stride 5, top-K 5, top-M 5, tolerance 3.
  These are C++ settings, not additional launch arguments.
- `Scancontext.h` retains `NUM_EXCLUDE_RECENT=50`; the sequence constructor passes
  recent-frame exclusion `8`. The offline search paths inherit these limits.
  They are **not equal searchable-map configurations by default**. Verify the
  search ranges used by every method before a controlled benchmark; packaging
  has not silently changed them to a different experiment configuration.
- `seq_type=1` selects the supplied legacy branch; it is not documented as a
  pure switch that changes only ERA's rate while preserving every other decision.
- `voxel_leaf_size` and `ndt_resolution` are legacy parameters. The current final
  registration call uses ICP; the name `ndt` in some local variables does not make
  it NDT. Read the call sites before treating a legacy parameter as an active control.
- Candidate-pose publishers remain as diagnostic interfaces, but their expensive
  candidate-registration call is commented out in the supplied single-frame path.
- The optional non-neighbor precompute currently uses the default height channel
  (`SC_CHANNEL=1`) and does not read `sc_channel`. Do not reuse its table for
  an intensity-channel run without adding a matching channel setting.

## Preprocessing conventions

The NTU converter preserves XYZ and stamps, outputs the original 96-byte radar
field layout, and validates required channel lengths. It does not infer absent
Doppler measurements. The ZF preprocessing preserves its original coordinate
flips, installation rotation and velocity model. Installation translations in
that model are not a general point-cloud-to-map transform.

Static and dynamic output clouds retain the input timestamp. Dynamic outputs are
RANSAC outliers; they do not also include every static point removed by the
unstable filter. `input_leaf_size` is retained but the supplied preprocessing
does not apply an additional voxel filter through this parameter.

## Validation scope

The release was assembled on Windows without a ROS Noetic/PCL toolchain. Local
checks can validate source consistency, XML, Python syntax/numerics and file
references; they cannot establish that a new Ubuntu build or dataset replay
has passed. Capture `scripts/print_environment.sh` output and run the checks in
the README on the deployment machine before claiming a tested release.

Checks completed for this copy:

- All 52 retained source-snapshot files still match their original SHA-256 hashes
  in the source directory; the original package was not edited.
- Descriptor, sequence alignment, preprocessing, and baseline implementation
  files match the source after package include-path substitution. The two core
  node constructors additionally replace the fixed log path with an optional parameter.
- All Python files pass Python 3.8 syntax parsing. Launch XML, include arguments,
  executable names, resource paths and local C++ header references are consistent.
- All 15 tests in `test_localization_recall.py` pass. Stale plot-label/grid-mode
  assertions were updated to the supplied plotting implementation. This local
  check used a Windows Python/Matplotlib runtime, not the target Focal stack.
- No navigation control topics, machine-specific runtime paths, generated data,
  Python caches or Windows line endings remain in the release tree.

ROS/C++ compilation, ROS integration tests, SciPy-dependent histogram tests and
bag replay have not been run in this environment.
