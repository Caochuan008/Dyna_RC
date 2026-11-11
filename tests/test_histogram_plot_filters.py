"""Plot-filter boundary and integration checks, without a ROS master."""
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
from matplotlib.figure import Figure

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/localization_error_histograms.py"
spec = importlib.util.spec_from_file_location("filtered_histograms", SCRIPT)
hist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hist)


class PlotFilterTests(unittest.TestCase):
    def test_near_zero_requires_both_strict_inequalities(self):
        samples = [(0.2, x, a) for x, a in [(0, 0), (0.009, 0.49), (0.01, 0.49),
                                           (0.009, 0.5), (0.005, 1), (1, 0.1)]]
        np.testing.assert_array_equal(hist.plot_sample_mask(samples), [False, False, True, True, True, True])
        self.assertTrue(hist.plot_sample_mask(samples, exclude_xy_below_m=0).all())
        self.assertTrue(hist.plot_sample_mask(samples, exclude_yaw_below_deg=0).all())

    def test_invalid_filter_settings(self):
        for settings in [(0, 0.01, 0.5), (1.5, 0.01, 0.5), (np.nan, 0.01, 0.5),
                         (100, -0.01, 0.5), (100, 0.01, np.inf)]:
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "Plot filters"):
                hist.validate_plot_filters(*settings)

    def test_99_hidden_100_visible_and_lines_break_in_all_plot_types(self):
        samples = [(0.2, 0.2, 1)] * 100 + [(0.5, 1.2, 1)] * 99 + [(0.8, 2.2, 1)] * 100
        success = [0, 1] * 50 + [1] * 99 + [0] * 100
        neighbors = [0.1] * 100 + [0.4] * 99 + [0.7] * 100
        # These excluded values must not extend axes or alter counts, means or success rates.
        samples += [(1.9, 0.001, 0.1)] * 30
        success += [1] * 30
        neighbors += [2.0] * 30
        figures = {}

        def capture(fig, path, **kwargs):
            if Path(path).suffix == ".png":
                figures[Path(path).stem] = fig

        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture):
            root = Path(directory)
            groups = hist.plot_histograms(samples, root, 3, 3, success, neighbors)
            self.assertEqual(groups["01_score_xy_yaw_lt5"], 299)
            self.assertEqual(len(figures), 18)
            ax = figures["01_score_xy_yaw_lt5"].axes[0]
            cells = ax.collections[0].get_array()
            np.testing.assert_array_equal(cells.compressed(), [100, 100])
            np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.2, np.nan, 0.8], equal_nan=True)
            for name in ("07_success_rate_xy_yaw_lt5", "13_non_neighbor_success_xy_yaw_lt5"):
                ax = figures[name].axes[0]
                np.testing.assert_allclose(ax.lines[0].get_ydata(), [50, np.nan, 0], equal_nan=True)
                with (root / (name + ".csv")).open() as file:
                    bins = list(csv.DictReader(file))
                self.assertEqual([int(row["count"]) for row in bins], [100, 99, 100])
                self.assertEqual([row["plotted"] for row in bins], ["1", "0", "1"])
                self.assertEqual(int(bins[1]["successes"]), 99)  # Hidden data is still exported.
            ax = figures["13_non_neighbor_success_xy_yaw_lt5"].axes[0]
            self.assertEqual(len(ax.collections[0].get_offsets()), 2)
            np.testing.assert_array_equal(ax.collections[0].get_array(), [100, 100])
            with np.load(root / "histogram_counts.npz") as counts:
                self.assertEqual(counts["01_score_xy_yaw_lt5"].sum(), 299)
                self.assertEqual(counts["score_edges"][-1], 0.8)
            # Exercise the real renderer, including the shared logarithmic colorbars.
            figures["01_score_xy_yaw_lt5"].canvas.draw()
            figures["13_non_neighbor_success_xy_yaw_lt5"].canvas.draw()

    def test_filtering_precedes_bin_count_threshold(self):
        samples = [(0.2, 0.005, 0.1)] + [(0.2, 0.5, 1)] * 99
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig"):
            hist.plot_histograms(samples, directory, 1, 1, success=[1] * 100)
            with (Path(directory) / "07_success_rate_xy_yaw_lt5.csv").open() as file:
                row = next(csv.DictReader(file))
            self.assertEqual((row["count"], row["plotted"]), ("99", "0"))

    def test_replot_loads_saved_filters_and_accepts_overrides_without_modifying_input(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig"):
            root = Path(directory)
            source = root / "observations.csv"
            with source.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("score", "xy_error_m", "yaw_error_deg", "localization_xy_error_m", "localization_yaw_error_deg"))
                writer.writerows([(0.2, 0.005, 0.4, 0, 0), (0.4, 0.02, 0.4, 0, 0),
                                  (0.6, 0.02, 0.4, 0, 0), (0.8, 0.02, 0.6, 0, 0)])
            original = source.read_bytes()
            (root / "summary.json").write_text(json.dumps(dict(min_bin_count=2, exclude_xy_below_m=0.03,
                                                              exclude_yaw_below_deg=0.5)))
            output = hist.replot_csv(source, score_bins=2, error_bins=2)
            settings = json.loads((output / "plot_settings.json").read_text())
            self.assertEqual((settings["min_bin_count"], settings["plot_sample_count"], settings["plot_excluded_near_zero"]), (2, 1, 3))
            hist.replot_csv(source, score_bins=2, error_bins=2, min_bin_count=1, exclude_xy_below_m=0.01)
            settings = json.loads((output / "plot_settings.json").read_text())
            self.assertEqual((settings["min_bin_count"], settings["plot_sample_count"], settings["plot_excluded_near_zero"]), (1, 3, 1))
            self.assertEqual(source.read_bytes(), original)

    def test_all_excluded_still_produces_empty_plots(self):
        def draw(fig, path, **kwargs):
            if Path(path).name == "13_non_neighbor_success_xy_yaw_lt5.png":
                fig.canvas.draw()
                self.assertEqual(len(fig.axes[0].collections[0].get_offsets()), 0)
                self.assertIn("No bins", fig.axes[0].texts[0].get_text())

        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", draw):
            counts = hist.plot_histograms([(0.1, 0, 0)], directory, 2, 2, [True], [0.2])
            self.assertEqual(sum(counts.values()), 0)

    def test_ros_parameters_are_cached_and_raw_near_zero_observation_is_retained(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig"):
            root = Path(directory)
            np.savetxt(root / "map.txt", [[0, 0, 0, 0, 0, 0, 0, 1]])
            np.savetxt(root / "gt.txt", [[0, 0.005, 0, 0, 0, 0, 0, 1], [1, 0.005, 0, 0, 0, 0, 0, 1]])
            params = {"~keyframe_poses_file": str(root / "map.txt"), "~gt_file": str(root / "gt.txt"),
                      "~output_dir": str(root), "~min_bin_count": 7, "~exclude_xy_below_m": 0.02,
                      "~exclude_yaw_below_deg": 1.0, "~score_bins": 2, "~error_bins": 2}
            ros = types.ModuleType("rospy")
            ros.get_param = lambda name, default=None: params.get(name, default)
            for name in ("loginfo", "logwarn", "loginfo_throttle", "logwarn_throttle", "logerr", "on_shutdown", "Subscriber"):
                setattr(ros, name, lambda *args, **kwargs: None)
            messages = types.ModuleType("nav_msgs.msg")
            messages.Odometry = object
            output_pose = types.SimpleNamespace(position=types.SimpleNamespace(x=10, y=0),
                orientation=types.SimpleNamespace(x=0, y=0, z=0, w=1))
            message = types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(to_sec=lambda: 0.5)),
                pose=types.SimpleNamespace(covariance=[0.2, 0] + [0] * 34, pose=output_pose))
            with patch.dict(sys.modules, {"rospy": ros, "nav_msgs": types.ModuleType("nav_msgs"), "nav_msgs.msg": messages}):
                node = hist.LocalizationHistograms()
                node.callback(message)
                ros.get_param = lambda *args: self.fail("Parameters must be cached before shutdown")
                node.save()
            settings = json.loads((node.output_dir / "summary.json").read_text())
            self.assertEqual((settings["min_bin_count"], settings["exclude_xy_below_m"], settings["exclude_yaw_below_deg"]), (7, 0.02, 1.0))
            self.assertEqual((settings["valid"], settings["plot_sample_count"], settings["plot_excluded_near_zero"]), (1, 0, 1))
            with (node.output_dir / "observations.csv").open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)


if __name__ == "__main__":
    unittest.main()
