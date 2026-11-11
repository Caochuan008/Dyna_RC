"""Numerical evaluation and CSV round-trip checks; no ROS master required."""
import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np


def load_module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recorder = load_module("localization_recall_recorder")
comparison = load_module("localization_recall_comparison")


def pose_row(stamp, x=0., z=0., angle=0.):
    a = np.radians(angle) / 2
    return [stamp, x, 0, z, 0, 0, np.sin(a), np.cos(a)]


def message(stamp_ns=1500000000, index=1., score=0.2):
    ns = types.SimpleNamespace
    return ns(header=ns(stamp=ns(to_nsec=lambda: stamp_ns, to_sec=lambda: stamp_ns / 1e9)),
              pose=ns(covariance=[score, index], pose=ns(position=ns(x=1.2, y=0., z=1.),
                         orientation=ns(x=0., y=0., z=0., w=1.))))


def run(label, records, invalid=()):
    return dict(label=label, records=records, invalid_gt=set(invalid), metadata={})


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.map = np.array([pose_row(2, 0), pose_row(1, 10, 3)])
        self.gt = recorder.GroundTruth(np.array([pose_row(1, 0, 1), pose_row(2, 2, 1)]))

    def test_matched_index_and_localized_pose_have_distinct_errors(self):
        row = recorder.make_record(message(), self.map, self.gt)
        self.assertEqual(row["matched_xy_error_m"], 9)
        self.assertAlmostEqual(row["matched_xyz_error_m"], np.sqrt(85))
        self.assertAlmostEqual(row["localization_xy_error_m"], .2)
        self.assertEqual(row["gt_x"], 1)
        self.assertEqual(row["matched_index_valid"], 1)

    def test_yaw_interpolation_takes_short_arc(self):
        gt = recorder.GroundTruth(np.array([pose_row(1, angle=179), pose_row(2, angle=-179)]))
        position, quaternion = gt.interpolate(1.5)
        self.assertAlmostEqual(abs(np.degrees(recorder.yaw(quaternion))), 180)
        np.testing.assert_allclose(position, [0, 0, 0])
        with self.assertRaisesRegex(ValueError, "outside_gt"):
            gt.interpolate(3)
        gt.max_gap = .5
        with self.assertRaisesRegex(ValueError, "gap"):
            gt.interpolate(1.5)

    def test_invalid_index_and_gt_do_not_discard_raw_observation(self):
        row = recorder.make_record(message(index=1.5), self.map, self.gt)
        self.assertTrue(np.isnan(row["matched_xy_error_m"]))
        self.assertEqual(row["gt_valid"], 1)
        self.assertIn("invalid_matched_index", row["status"])
        stamp = 1791472388111469984
        row = recorder.make_record(message(stamp), self.map, self.gt)
        self.assertEqual(row["stamp_ns"], stamp)
        self.assertEqual(row["gt_valid"], 0)
        self.assertEqual(row["score"], .2)

    def test_load_map_preserves_index_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.txt"
            np.savetxt(path, self.map)
            np.testing.assert_allclose(recorder.load_poses(path), self.map)

    def test_fixed_denominator_missing_outputs_and_strict_boundaries(self):
        runs = [run("A", {1: (.2, .5), 2: (.4, 1), 4: (.1, np.nan)}),
                run("B", {1: (.1, .2), 3: (.1, .2)})]
        curves, queries, total = comparison.evaluate(runs, [.2, .4, 1], [1], "fixed")
        self.assertEqual(queries, {1, 2, 3, 4})
        self.assertEqual(total, 4)
        self.assertEqual([r["success_count"] for r in curves[:3]], [0, 1, 1])
        self.assertEqual([r["success_rate_pct"] for r in curves[:3]], [0, 25, 25])
        self.assertEqual([r["success_rate_pct"] for r in curves[3:]], [50, 50, 50])
        self.assertEqual(curves[0]["accepted_count"], 1)  # Invalid error is still an accepted failure.
        extended, _, _ = comparison.evaluate(runs, [1], [1], "fixed", total_queries=8)
        self.assertEqual(extended[0]["success_rate_pct"], 12.5)
        with self.assertRaises(ValueError):
            comparison.evaluate(runs, [1], [1], "fixed", total_queries=3)

    def test_success_fraction_uses_each_algorithms_accepted_results(self):
        samples = {i: (.1, .5 if i < 80 else 1.) for i in range(100)}
        samples.update({i: (.2, 3.) for i in range(100, 140)})
        runs = [run("A", samples), run("B", {i: (.1, .5) for i in range(10)})]
        curves, _, _ = comparison.evaluate(runs, [0., .2, .22], [1.])
        self.assertTrue(np.isnan(curves[0]["success_rate_pct"]))
        self.assertEqual(curves[1]["denominator_count"], 100)
        self.assertEqual(curves[1]["success_count"], 80)
        self.assertEqual(curves[1]["success_rate_pct"], 80.)
        self.assertAlmostEqual(curves[2]["success_rate_pct"], 100. * 80 / 140)
        self.assertEqual(curves[4]["denominator_count"], 10)
        self.assertEqual(curves[4]["success_rate_pct"], 100.)

    def test_accepted_accuracy_and_gt_consistency(self):
        runs = [run("A", {1: (.2, .5), 2: (.4, 2)})]
        curves, _, _ = comparison.evaluate(runs, [0, .3, 1], [1], "accepted")
        self.assertTrue(np.isnan(curves[0]["success_rate_pct"]))
        self.assertEqual([r["success_rate_pct"] for r in curves[1:]], [100, 50])
        with self.assertRaises(ValueError):
            comparison.evaluate(runs + [run("B", {}, invalid=[1])], [1], [1])

    def test_decimal_score_grid_matches_exact_threshold(self):
        grid = comparison.score_grid(0., 1., .02)
        self.assertEqual(len(grid), 51)
        self.assertEqual(grid[10], .2)
        curves, _, _ = comparison.evaluate([run("A", {1: (.2, .5)})], grid, [1])
        self.assertEqual(curves[10]["success_count"], 0)
        self.assertEqual(curves[11]["success_count"], 1)
        np.testing.assert_allclose(comparison.score_grid(0., 1., .3), [0, .3, .6, .9])
        grid = comparison.score_grid(.1, .6, .02)
        self.assertEqual((len(grid), grid[0], grid[-1]), (26, .1, .6))
        np.testing.assert_allclose(np.diff(grid), .02)

    def test_acceptance_uses_all_csv_rows_and_each_algorithms_own_total(self):
        # Includes a score below the plotted range, a strict boundary, and invalid scores.
        runs = [dict(label="A", all_scores=[.05, .1, .2, np.nan, -1.]),
                dict(label="B", all_scores=[.15, .3]), dict(label="Empty", all_scores=[])]
        rows = comparison.evaluate_acceptance(runs, [.1, .2, .6])
        self.assertEqual([r["score_accepted_count"] for r in rows[:3]], [1, 2, 3])
        self.assertEqual([r["accepted_fraction_pct"] for r in rows[:3]], [20., 40., 60.])
        self.assertEqual([r["accepted_fraction_pct"] for r in rows[3:6]], [0., 50., 100.])
        self.assertTrue(all(np.isnan(r["accepted_fraction_pct"]) for r in rows[6:]))

    def test_recall_vs_acceptance_pairs_algorithm_and_score_without_changing_denominators(self):
        # A has a score-passing output without GT; B has a different number of outputs.
        a = run("A", {1: (.05, 1.), 2: (.2, 4.), 3: (.4, 8.)}, invalid=[4])
        b = run("B", {1: (.1, 3.), 2: (.3, 9.)})
        a["all_scores"], b["all_scores"] = [.05, .2, .4, .08], [.1, .3]
        curves, _, _ = comparison.evaluate([a, b], [0., .2, .3, .5], [2., 5., 10.])
        acceptance = comparison.evaluate_acceptance([a, b], [0., .2, .3, .5])
        paired = comparison.pair_acceptance_recall(curves, list(reversed(acceptance)))
        self.assertEqual(len(paired), len(curves))
        for original, current in zip(curves, paired):
            for key in ("algorithm", "sc_dist_thres", "recall_distance_m", "success_count",
                        "accepted_count", "denominator_count"):
                self.assertEqual(current[key], original[key])
            np.testing.assert_allclose(current["success_rate_pct"], original["success_rate_pct"])
        selected = {(row["algorithm"], row["recall_distance_m"], row["sc_dist_thres"]): row for row in paired}
        point = selected["A", 2., .2]
        self.assertEqual((point["accepted_fraction_pct"], point["success_rate_pct"]), (50., 100.))
        self.assertEqual((point["score_accepted_count"], point["accepted_count"]), (2, 1))
        self.assertEqual(selected["B", 2., .2]["success_rate_pct"], 0.)
        self.assertEqual(selected["B", 5., .2]["success_rate_pct"], 100.)
        self.assertEqual(selected["B", 5., .2]["accepted_fraction_pct"], 50.)
        self.assertAlmostEqual(selected["A", 5., .5]["success_rate_pct"], 200. / 3.)
        self.assertEqual(selected["A", 5., 0.]["accepted_fraction_pct"], 0.)
        self.assertTrue(np.isnan(selected["A", 5., 0.]["success_rate_pct"]))

    def test_metadata_prevents_extrapolating_recorded_score_gate(self):
        item = run("A", {})
        item["metadata"] = {"producer_sc_dist_thres": .5}
        with self.assertRaisesRegex(ValueError, "recorded sc_dist_thres"):
            comparison.validate_runs([item], 1, "matched")

    def test_native_raplace_score_range_keeps_metric_denominators(self):
        sc = run("SC", {1: (.1, 1.), 2: (.4, 6.)})
        rp = run("RaPlace", {1: (.8, 1.), 2: (1.2, 6.)})
        sc.update(all_scores=[.1, .4], score_max=.98, score_thresholds=[0., .2, .98])
        rp.update(all_scores=[.8, 1.2], score_max=2., score_thresholds=[0., 1., 1.2, 2.])
        sc["metadata"]["producer_sc_dist_thres"] = .99
        rp["metadata"]["producer_sc_dist_thres"] = 1e6
        comparison.validate_runs([sc, rp], 2., "localization")
        curves, _, _ = comparison.evaluate([sc, rp], [0., .98], [2., 5., 10.])
        accepted = comparison.evaluate_acceptance([sc, rp], [0., .98])
        paired = comparison.pair_acceptance_recall(curves, accepted)
        points = {(p["algorithm"], p["recall_distance_m"], p["sc_dist_thres"]): p for p in paired}
        self.assertEqual(len(curves), 3 * (3 + 4))
        self.assertEqual(points["RaPlace", 2., 1.2]["accepted_count"], 1)  # Strict score gate.
        self.assertEqual(points["RaPlace", 2., 2.]["success_rate_pct"], 50.)
        self.assertEqual(points["RaPlace", 2., 2.]["accepted_fraction_pct"], 100.)
        self.assertEqual(points["RaPlace", 10., 2.]["success_rate_pct"], 100.)
        rp["metadata"]["producer_sc_dist_thres"] = 1.
        with self.assertRaisesRegex(ValueError, "recorded sc_dist_thres"):
            comparison.validate_runs([sc, rp], .98, "localization")

    def test_four_algorithms_can_plot_different_native_score_ranges(self):
        from matplotlib.figure import Figure
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(1, 5):
                with (root / ("alg%d.csv" % i)).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=recorder.FIELDS); writer.writeheader()
                    writer.writerow(recorder.make_record(message(score=1.2 if i == 4 else .2), self.map, self.gt))
                (root / ("alg%d.meta.json" % i)).write_text(json.dumps(dict(
                    producer_sc_dist_thres=1e6 if i == 4 else .99)))
            captured = []
            def capture(fig, path, **kwargs):
                if Path(path).name == "recall_vs_acceptance.png": captured.append(fig)
            with patch.object(Figure, "savefig", capture):
                comparison.compare(dict(csv_dir=str(root), algorithm_count=4, score_max=.98,
                                        score_mode="grid", score_normalization="none",
                                        algorithm_score_maxima=[.98, .98, .98, 2.]))
            self.assertEqual(len(captured[0].axes), 3)
            for ax in captured[0].axes:
                self.assertEqual(len(ax.lines), 4)
                self.assertEqual([len(line.get_xdata()) for line in ax.lines], [50, 50, 50, 101])
                self.assertEqual(ax.lines[3].get_xdata()[-1], 100.)
                self.assertEqual(ax.lines[3].get_ydata()[-1], 100.)

    def test_recorder_csv_metadata_and_shutdown(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savetxt(root / "map.txt", self.map)
            np.savetxt(root / "gt.txt", self.gt.poses)
            params = {"~keyframe_poses_file": str(root / "map.txt"), "~gt_file": str(root / "gt.txt"),
                      "~output_csv": str(root / "custom.csv"), "~algorithm_name": "Test algorithm", "/sc_dist_thres": 1.}
            ros = types.ModuleType("rospy")
            ros.get_param = lambda key, default=None: params.get(key, default)
            for name in ("on_shutdown", "Subscriber", "loginfo", "logwarn", "logwarn_throttle", "loginfo_throttle"):
                setattr(ros, name, lambda *a, **k: None)
            msgs = types.ModuleType("nav_msgs.msg")
            msgs.Odometry = object
            with patch.dict(sys.modules, {"rospy": ros, "nav_msgs.msg": msgs}):
                node = recorder.RecallRecorder()
                node.callback(message())
                node.callback(message())
                node.callback(message(3000000000))
                node.close()
                node.close()
                loaded = comparison.load_run(root / "custom.csv", "matched_xy_error_m")
                self.assertEqual(loaded["label"], "Test algorithm")
                self.assertEqual(len(loaded["records"]), 1)
                self.assertEqual(len(loaded["invalid_gt"]), 1)
                # A row without GT is excluded from success evaluation, but still an output.
                accepted = comparison.evaluate_acceptance([loaded], [.3])[0]
                self.assertEqual(accepted["recorded_count"], 2)
                self.assertEqual(accepted["score_accepted_count"], 2)
                self.assertEqual(accepted["accepted_fraction_pct"], 100.)
                self.assertEqual(loaded["metadata"]["recorded"], 2)
                self.assertEqual(loaded["metadata"]["counters"]["duplicate_stamp"], 1)
                self.assertEqual(loaded["metadata"]["status"], "complete")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    recorder.RecallRecorder()
                # A sequence run uses its own gate, not the unrelated single-frame setting.
                params.update({"~output_csv": str(root / "sequence.csv"),
                               "~localization_topic": "/sequence/localization_odom",
                               "/seq_distance_threshold": 1., "/sc_dist_thres": .99})
                sequence = recorder.RecallRecorder()
                sequence.callback(message())
                sequence.close()
                loaded = comparison.load_run(root / "sequence.csv", "localization_xyz_error_m")
                self.assertEqual(loaded["metadata"]["producer_score_threshold_param"], "/seq_distance_threshold")
                comparison.validate_runs([loaded], 1., "localization")

    def test_paper_row_has_one_acceptance_three_recalls_and_readable_fonts(self):
        from matplotlib.figure import Figure
        figures = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(1, 4):
                with (root / ("alg%d.csv" % i)).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=recorder.FIELDS)
                    writer.writeheader()
                    observation = message(score=.1 * i)
                    observation.pose.pose.position.z = 5.  # XY succeeds at 2 m; XYZ would fail.
                    writer.writerow(recorder.make_record(observation, self.map, self.gt))
            def capture(fig, path, **kwargs):
                if Path(path).suffix == ".png":
                    figures.append((Path(path).stem, fig))
            with patch.object(Figure, "savefig", capture), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = comparison.compare(dict(csv_dir=str(root), algorithm_labels=["A", "B", "C"],
                                                 score_mode="grid", score_normalization="none",
                                                 score_min=.1, score_max=.6, score_step=.02))
            self.assertEqual([name for name, _ in figures],
                             ["comparison_row", "acceptance", "recall_2m", "recall_5m", "recall_10m",
                              "recall_vs_acceptance"])
            for name, fig in figures[:5]:
                self.assertEqual(len(fig.legends), 1)
                self.assertEqual([text.get_fontsize() for text in fig.legends[0].get_texts()], [9.] * 3)
                for ax in fig.axes:
                    np.testing.assert_allclose(ax.get_xlim(), [.1, .6])
                    self.assertEqual(ax.xaxis.label.get_fontsize(), 9.)
                    lines = ax.lines
                    self.assertEqual(len({line.get_color() for line in lines}), 3)
                    self.assertTrue(all(line.get_markeredgecolor() == line.get_color() for line in lines))
                    self.assertEqual(len({(line.get_marker(), line.get_linestyle()) for line in lines}), 3)
                    self.assertEqual([line.get_label() for line in lines], ["A", "B", "C"])
                    for line in lines:
                        self.assertAlmostEqual(line.get_ydata()[-1], 100.)
                        # Current plots space markers along the displayed axis.
                        self.assertEqual(line.get_markevery()[0], 0)
                        self.assertEqual(line.get_markevery()[-1], len(line.get_xdata()) - 1)
                        np.testing.assert_allclose(np.diff(line.get_xdata()), .02)
            row = figures[0][1]
            np.testing.assert_allclose(row.get_size_inches() * 25.4, [180., 58.])
            self.assertEqual([ax.get_title() for ax in row.axes],
                             ["(a) Accepted results", "(b) Recall@2 m", "(c) Recall@5 m", "(d) Recall@10 m"])
            self.assertEqual([ax.get_ylabel() for ax in row.axes], ["Accepted (%)"] + ["Localization success (%)"] * 3)
            for ax, (_, individual) in zip(row.axes, figures[1:5]):
                for first, current in zip(ax.lines, individual.axes[0].lines):
                    np.testing.assert_allclose(first.get_ydata(), current.get_ydata())
            tradeoff = figures[-1][1]
            self.assertEqual(len(tradeoff.axes), 3)
            self.assertEqual(len(tradeoff.legends), 1)
            self.assertEqual([text.get_fontsize() for text in tradeoff.legends[0].get_texts()], [9.] * 3)
            np.testing.assert_allclose(tradeoff.get_size_inches() * 25.4, [180., 58.])
            self.assertEqual([ax.get_title() for ax in tradeoff.axes],
                             ["(a) Recall@2 m", "(b) Recall@5 m", "(c) Recall@10 m"])
            for ax, previous in zip(tradeoff.axes, row.axes[1:]):
                self.assertEqual(ax.get_xlabel(), "Effective output rate (%)")
                np.testing.assert_allclose(ax.get_xlim(), [0., 100.])
                for line, accepted_line, recall_line in zip(ax.lines, row.axes[0].lines, previous.lines):
                    np.testing.assert_allclose(line.get_xdata(), accepted_line.get_ydata())
                    np.testing.assert_allclose(line.get_ydata(), recall_line.get_ydata())
                    self.assertEqual((line.get_color(), line.get_marker(), line.get_linestyle()),
                                     (recall_line.get_color(), recall_line.get_marker(), recall_line.get_linestyle()))
                    self.assertEqual(line.get_markevery()[0], 0)
                    self.assertEqual(line.get_xdata()[line.get_markevery()[-1]], max(line.get_xdata()))
            self.assertEqual(report["union_queries"], 1)
            self.assertEqual(report["error_column"], "localization_xy_error_m")
            with (root / "comparison/recall_curves.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3 * 3 * 26)
            with (root / "comparison/acceptance_curves.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3 * 26)
            with (root / "comparison/recall_vs_acceptance_curves.csv").open() as stream:
                paired_rows = list(csv.DictReader(stream))
                self.assertEqual(len(paired_rows), 3 * 3 * 26)
                self.assertIn("accepted_fraction_pct", paired_rows[0])
                self.assertIn("sc_dist_thres", paired_rows[0])
            self.assertTrue((root / "comparison/comparison_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
