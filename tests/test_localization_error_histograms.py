"""Offline numerical/recording checks; no ROS master or robot required."""
import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/localization_error_histograms.py"
spec = importlib.util.spec_from_file_location("histograms", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pose(stamp, x=0, y=0, angle=0):
    half = np.deg2rad(angle) / 2
    return [stamp, x, y, 0, 0, 0, np.sin(half), np.cos(half)]


def message(stamp, score=0.2, index=0, x=3, y=0, angle=-170):
    half = np.deg2rad(angle) / 2
    output_pose = types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y, z=0),
        orientation=types.SimpleNamespace(x=0, y=0, z=np.sin(half), w=np.cos(half)))
    return types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(to_sec=lambda: stamp)),
                                 pose=types.SimpleNamespace(covariance=[score, index] + [0] * 34, pose=output_pose))


def non_neighbor_table(keyframes, scores, confusers, counts):
    return np.column_stack((np.arange(len(keyframes)), keyframes[:, 1:8], scores, confusers, counts))


def write_non_neighbor_csv(path, table, xy_threshold=3, yaw_threshold_deg=15):
    with path.open("w", newline="") as file:
        file.write("# metric=non_neighbor_v1_min_cosine_dynamic\n# xy_threshold=%s\n# yaw_threshold_deg=%s\n" % (xy_threshold, yaw_threshold_deg))
        writer = csv.writer(file)
        writer.writerow(module.NON_NEIGHBOR_FIELDS)
        writer.writerows(table)


class HistogramTests(unittest.TestCase):
    def test_non_neighbor_geometry_uses_complement_of_success_region(self):
        # Same XY but 16-degree yaw difference is a non-neighbor; exactly 3 m/15 deg is a neighbor.
        keyframes = np.array([pose(0), pose(1, angle=16), pose(2, x=3, angle=15)])
        data = non_neighbor_table(keyframes, [0.1, 0.1, np.nan], [1, 0, -1], [1, 1, 0])
        table = module.validate_non_neighbor_table(data, keyframes)
        self.assertAlmostEqual(table[0, 8], 0.1)
        table[0, 9] = 2
        with self.assertRaisesRegex(ValueError, "geometrically valid neighbor"):
            module.validate_non_neighbor_table(table, keyframes)

    def test_non_neighbor_table_rejects_wrong_map_and_missing_score_zero_substitute(self):
        keyframes = np.array([pose(0), pose(1, x=10)])
        data = non_neighbor_table(keyframes, [0.2, np.nan], [1, -1], [1, 0])
        table = module.validate_non_neighbor_table(data, keyframes)
        wrong_map = keyframes.copy()
        wrong_map[0, 1] += 1
        with self.assertRaisesRegex(ValueError, "positions differ"):
            module.validate_non_neighbor_table(table, wrong_map)
        table[1, 8] = 0
        with self.assertRaisesRegex(ValueError, "Invalid non-neighbor"):
            module.validate_non_neighbor_table(table, keyframes)

    def test_offline_csv_loading_validates_metadata_and_never_needs_a_topic(self):
        keyframes = np.array([pose(0), pose(1, x=10)])
        data = non_neighbor_table(keyframes, [0.2, np.nan], [1, -1], [1, 0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.csv"
            write_non_neighbor_csv(path, data)
            table, metadata = module.load_non_neighbor_csv(path, keyframes)
            np.testing.assert_allclose(table, data, equal_nan=True)
            self.assertEqual(metadata["metric"], "non_neighbor_v1_min_cosine_dynamic")
            write_non_neighbor_csv(path, data, xy_threshold=5)
            self.assertEqual(module.load_non_neighbor_csv(path, keyframes)[1]["xy_threshold"], "5")
            for xy_threshold, yaw_threshold in ((-1, 15), (3, 181), (np.nan, 15)):
                write_non_neighbor_csv(path, data, xy_threshold, yaw_threshold)
                with self.assertRaisesRegex(ValueError, "Success thresholds"):
                    module.load_non_neighbor_csv(path, keyframes)
            # Both XY-only and yaw-only non-neighbors must be checked using the CSV's own thresholds.
            nearby = np.array([pose(0), pose(1, x=1.5), pose(2, angle=6)])
            custom = non_neighbor_table(nearby, [0.2, 0.2, 0.3], [1, 0, 0], [2, 2, 2])
            write_non_neighbor_csv(path, custom, 1, 5)
            np.testing.assert_allclose(module.load_non_neighbor_csv(path, nearby)[0], custom)
            write_non_neighbor_csv(path, custom, 3, 15)
            with self.assertRaisesRegex(ValueError, "geometrically valid neighbor"):
                module.load_non_neighbor_csv(path, nearby)
            with self.assertRaises(FileNotFoundError):
                module.load_non_neighbor_csv(Path(directory) / "missing.csv", keyframes)

    def test_non_neighbor_replot_bins_by_map_score_and_excludes_only_missing_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "observations.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("score", "xy_error_m", "yaw_error_deg", "localization_xy_error_m", "localization_yaw_error_deg", "non_neighbor_score", "success"))
                # Stored labels are deliberately wrong: replot must use raw pose errors and the overridden thresholds.
                writer.writerows([(0.9, 0.2, 0, 1, 5, 0.1, 0), (0.8, 0.8, 0, 1.1, 0, 0.1, 1),
                                  (0.1, 0.5, 20, 0, 0, 0.1, 0), (0.1, 4, 0, 0, 5.1, 0.4, 1),
                                  (0.2, 0.5, 0, 0, 0, np.nan, 0)])
            (root / "summary.json").write_text(json.dumps({"success_xy_m": 10, "success_yaw_deg": 180}))
            output = module.replot_csv(root / "observations.csv", score_bins=2, error_bins=4, success_xy_m=1, success_yaw_deg=5)
            self.assertEqual(len(list(output.glob("*.png"))), 18)
            self.assertEqual(len(list(output.glob("*.svg"))), 18)
            with (output / "13_non_neighbor_success_xy_yaw_lt5.csv").open() as file:
                bins = list(csv.DictReader(file))
            self.assertEqual(len(bins), 2)
            self.assertAlmostEqual(float(bins[0]["success_rate"]), 0.5)
            self.assertEqual([int(row["count"]) for row in bins], [2, 1])
            self.assertEqual(float(bins[-1]["success_rate"]), 0)
            with (output / "07_success_rate_xy_yaw_lt5.csv").open() as file:
                success_bins = list(csv.DictReader(file))
            self.assertEqual(sum(int(row["successes"]) for row in success_bins), 2)
            self.assertEqual(sum(int(row["count"]) for row in success_bins), 4)
            settings = json.loads((output / "plot_settings.json").read_text())
            self.assertEqual((settings["success_xy_m"], settings["success_yaw_deg"]), (1, 5))

    def test_success_requires_both_thresholds_and_includes_boundaries(self):
        xy = np.array([3, 3.000001, 0, 0, np.nan, -1])
        angle = np.array([15, 0, 15.000001, 0, 0, 0])
        np.testing.assert_array_equal(module.success_from_errors(xy, angle), [True, False, False, True, False, False])
        np.testing.assert_array_equal(module.success_from_errors([1, 1.001, 0, 0], [5, 0, 5.001, 0], 1, 5),
                                      [True, False, False, True])
        for thresholds in ((-1, 15), (3, 181), (np.nan, 15)):
            with self.assertRaises(ValueError):
                module.success_from_errors(0, 0, *thresholds)

    def test_output_pose_error_wraps_yaw_and_invalid_quaternion_fails(self):
        output_pose = message(0, x=6, y=4, angle=-175).pose.pose
        metrics = module.localization_metrics(output_pose, np.array([3, 4, 50]), np.deg2rad(170))
        np.testing.assert_allclose(metrics[-2:], [3, 15], atol=1e-10)
        self.assertTrue(module.success_from_errors(*metrics[-2:]))
        output_pose.orientation = types.SimpleNamespace(x=0, y=0, z=0, w=0)
        metrics = module.localization_metrics(output_pose, np.array([3, 4, 50]), np.deg2rad(170))
        self.assertFalse(module.success_from_errors(*metrics[-2:]))

    def test_success_replot_uses_output_errors_and_shared_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "observations.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("score", "xy_error_m", "yaw_error_deg", "localization_xy_error_m", "localization_yaw_error_deg"))
                writer.writerows([(0.2, 0.2, 0, 1, 5), (0.2, 0.8, 0, 1.1, 0),
                                  (0.2, 0.5, 20, 0, 0), (0.2, 4, 0, 0, 5.1)])
            (root / "summary.json").write_text(json.dumps({"success_xy_m": 1, "success_yaw_deg": 5}))
            output = module.replot_csv(root / "observations.csv", error_bins=4)
            self.assertEqual(len(list(output.glob("*.png"))), 12)
            self.assertEqual(len(list(output.glob("*.svg"))), 12)
            with (output / "07_success_rate_xy_yaw_lt5.csv").open() as file:
                bins = list(csv.DictReader(file))
            self.assertEqual(len(bins), 4)
            self.assertAlmostEqual(float(bins[0]["success_rate"]), 0.5)
            self.assertEqual((int(bins[0]["count"]), int(bins[0]["successes"])), (2, 1))
            self.assertTrue(np.isnan(float(bins[1]["success_rate"])))
            self.assertEqual(float(bins[-1]["success_rate"]), 0)
            with (output / "01_score_xy_yaw_lt5_mean.csv").open() as file:
                means = list(csv.DictReader(file))
            self.assertEqual([row["error_center"] for row in bins], [row["error_center"] for row in means])

    def test_score_means_use_raw_values_and_keep_empty_bins(self):
        # Interior boundaries belong to the next bin; the final upper edge is included.
        means, counts = module.mean_score_by_error(
            np.array([0.11, 0.39, 0.7, 0.9]), np.array([0, 0.4, 1, 4]), np.array([0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(counts, [2, 1, 0, 1])
        np.testing.assert_allclose(means, [0.25, 0.7, np.nan, 0.9], equal_nan=True)

    def test_csv_replot_preserves_conditions_and_saved_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "observations.csv"
            with csv_path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("score", "xy_error_m", "yaw_error_deg"))
                writer.writerows([(0.11, 0.2, 0), (0.39, 0.8, 5), (0.9, 0.5, 20), (0.8, 4, 0)])
            (root / "summary.json").write_text(json.dumps({"score_bins": 4, "error_bins": 4,
                                                           "yaw_group_low_deg": 6, "yaw_group_high_deg": 18}))
            with self.assertWarnsRegex(RuntimeWarning, "Old CSV"):
                output = module.replot_csv(csv_path)
            self.assertEqual(output, root / "with_mean")
            self.assertEqual(len(list(output.glob("*success_rate*.png"))), 0)
            with (output / "01_score_xy_yaw_lt6_mean.csv").open() as file:
                means = list(csv.DictReader(file))
            self.assertEqual(len(means), 4)
            self.assertAlmostEqual(float(means[0]["mean_score"]), 0.25)
            self.assertEqual(int(means[0]["count"]), 2)  # Excludes the 20-degree sample.
            self.assertTrue(np.isnan(float(means[1]["mean_score"])))
            self.assertAlmostEqual(float(means[-1]["mean_score"]), 0.8)

    def test_translation_and_shortest_rotation_interpolation(self):
        gt = module.GroundTruth(np.array([pose(100, 0, 2, 170), pose(102, 4, 6, -170)]), 3)
        position, angle = gt.interpolate(101)
        np.testing.assert_allclose(position, [2, 4, 0])
        self.assertAlmostEqual(abs(np.rad2deg(angle)), 180)
        self.assertAlmostEqual(np.rad2deg(gt.interpolate(102)[1]), -170)

    def test_equal_rotation_opposite_quaternion_signs(self):
        poses = np.array([pose(0, angle=25), pose(1, angle=25)])
        poses[1, 4:8] *= -1
        gt = module.GroundTruth(poses, 1)
        self.assertAlmostEqual(np.rad2deg(gt.interpolate(0.5)[1]), 25)

    def test_no_extrapolation_and_no_crossing_large_gaps(self):
        gt = module.GroundTruth(np.array([pose(1), pose(10)]), 1)
        for stamp in (0, 11):
            with self.assertRaisesRegex(ValueError, "outside_gt_time_range"):
                gt.interpolate(stamp)
        with self.assertRaisesRegex(ValueError, "gt_interpolation_gap"):
            gt.interpolate(5)
        np.testing.assert_allclose(gt.interpolate(1)[0], [0, 0, 0])
        gt.max_gap = 0
        np.testing.assert_allclose(gt.interpolate(5)[0], [0, 0, 0])

    def test_duplicate_gt_and_sorting(self):
        gt = module.GroundTruth(np.array([pose(2), pose(1), pose(1)]), 1)
        np.testing.assert_array_equal(gt.times, [1, 2])
        with self.assertRaisesRegex(ValueError, "Conflicting GT"):
            module.GroundTruth(np.array([pose(1), pose(1, x=2)]), 1)

    def test_loading_keeps_keyframe_indices_and_rejects_bad_quaternions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.txt"
            np.savetxt(path, [pose(2, x=20), pose(1, x=10)])
            np.testing.assert_array_equal(module.load_poses(path)[:, 1], [20, 10])
            invalid = np.array([pose(0)])
            invalid[:, 4:8] = 0
            np.savetxt(path, invalid)
            with self.assertRaisesRegex(ValueError, "Zero quaternion"):
                module.load_poses(path)

    def test_histogram_boundaries_and_total_counts(self):
        # Both 5 and 15 degrees belong to the middle group; values above 15 do not.
        samples = [(0.2, distance, angle) for distance in (0, 1, 3) for angle in (0, 5, 15, 15.001)]
        with tempfile.TemporaryDirectory() as directory:
            counts = module.plot_histograms(samples, directory, 5, 5)
            expected_counts = [2, 6, 3, 3, 4, 4]  # The jointly zero XY/yaw sample is excluded from plots.
            self.assertEqual(list(counts.values()), expected_counts)
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 6)
            self.assertEqual(len(list(Path(directory).glob("*.svg"))), 6)
            with np.load(Path(directory) / "histogram_counts.npz") as saved:
                for name, expected in zip(counts, expected_counts):
                    self.assertEqual(int(saved[name].sum()), expected)

    def test_empty_run_still_writes_twelve_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = module.plot_histograms([], directory, 2, 2, success=[])
            self.assertEqual(sum(groups.values()), 0)
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 12)

    def test_recording_uses_gt_nearest_keyframe_and_shutdown_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "map.txt", [pose(0, x=0, angle=170), pose(1, x=100)])
            np.savetxt(root / "gt.txt", [pose(10, x=2, angle=-170), pose(11, x=4, angle=-170)])
            write_non_neighbor_csv(root / "lookup.csv", non_neighbor_table(module.load_poses(root / "map.txt"), [0.1, 0.8], [1, 0], [1, 1]))
            params = {"~keyframe_poses_file": str(root / "map.txt"), "~gt_file": str(root / "gt.txt"),
                      "~output_dir": str(root), "~gt_time_offset": 10, "~score_bins": 3, "~error_bins": 3,
                      "~non_neighbor_csv": str(root / "lookup.csv"), "~yaw_group_low_deg": 10, "~yaw_group_high_deg": 30,
                      "~success_xy_m": 1, "~success_yaw_deg": 5}
            ros = types.ModuleType("rospy")
            ros.get_param = lambda name, default=None: params.get(name, default)
            for name in ("loginfo", "logwarn", "loginfo_throttle", "logwarn_throttle", "logerr", "on_shutdown"):
                setattr(ros, name, lambda *args: None)
            topics = []
            ros.Subscriber = lambda topic, *args, **kwargs: topics.append(topic)
            messages = types.ModuleType("nav_msgs.msg")
            messages.Odometry = object
            with patch.dict(sys.modules, {"rospy": ros, "nav_msgs": types.ModuleType("nav_msgs"), "nav_msgs.msg": messages}):
                node = module.LocalizationHistograms()
                self.assertEqual(topics, ["/localization_odom"])
                # The deliberately distant matched index must not control the geometric error.
                node.callback(message(0.5, index=1))
                np.testing.assert_allclose(node.samples, [[0.2, 3, 20]], atol=1e-10)
                self.assertEqual(node.non_neighbor_scores, [0.1])  # Loaded before the first message; matched index is 1.
                self.assertEqual(node.success, [True])  # Nearest-keyframe yaw error is NOT the success criterion.
                node.callback(message(0.5, x=4.1))
                node.callback(message(0.5, angle=-164))
                node.callback(message(0.5, angle=np.nan))  # Kept in the denominator as a failure.
                self.assertEqual(node.success, [True, False, False, False])
                node.callback(message(2))  # GT time range miss.
                node.callback(message(0.5, score=float("nan")))
                node.callback(message(0.5, index=1.9999))  # Rounded index is out of range.
                self.assertIsNotNone(node.non_neighbor_table)
                # ROS master can be gone during shutdown: save must use cached parameters.
                ros.get_param = lambda *args: (_ for _ in ()).throw(RuntimeError("master unavailable"))
                node.save()
                node.save()
                node.callback(message(0.5))
            summary = json.loads((node.output_dir / "summary.json").read_text())
            self.assertEqual((summary["received"], summary["valid"]), (7, 4))
            self.assertEqual(sum(summary["skipped"].values()), 3)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(summary["success_rate"], 0.25)
            self.assertEqual((summary["success_xy_m"], summary["success_yaw_deg"]), (1, 5))
            self.assertEqual(summary["invalid_localization_poses"], 1)
            self.assertEqual(summary["non_neighbor_valid_samples"], 4)
            self.assertEqual((summary["yaw_group_low_deg"], summary["yaw_group_high_deg"]), (10, 30))
            self.assertIn("02_score_xy_yaw_10to30", summary["groups"])
            with (node.output_dir / "observations.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual((rows[0]["matched_index"], rows[0]["nearest_index"]), ("1", "0"))
            self.assertEqual([row["success"] for row in rows], ["1", "0", "0", "0"])
            self.assertAlmostEqual(float(rows[0]["localization_xy_error_m"]), 0)
            self.assertTrue(all(float(row["non_neighbor_score"]) == 0.1 for row in rows))
            self.assertTrue(all(row["non_neighbor_index"] == "1" for row in rows))


if __name__ == "__main__":
    unittest.main()
