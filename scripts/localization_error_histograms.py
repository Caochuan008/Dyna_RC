#!/usr/bin/env python3
"""Plot localization scores and success rates versus GT-to-nearest-keyframe offsets."""
import csv
import argparse
import copy
import json
import os
import shutil
import threading
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree

SUCCESS_XY_M, SUCCESS_YAW_DEG = 8.0, 15.0
NON_NEIGHBOR_FIELDS = ("index", "x", "y", "z", "qx", "qy", "qz", "qw", "score", "confuser_index", "valid_comparisons")


def load_poses(path):
    """Columns: timestamp x y z qx qy qz qw; preserve map row order as indices."""
    poses = np.loadtxt(path, comments="#", usecols=range(8), ndmin=2)
    if not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Empty or non-finite trajectory: " + str(path))
    norms = np.linalg.norm(poses[:, 4:8], axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Zero quaternion in " + str(path))
    poses[:, 4:8] /= norms[:, None]
    return poses


def yaw(quaternion):
    x, y, z, w = np.asarray(quaternion).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_error_deg(a, b):
    return float(np.degrees(abs(np.arctan2(np.sin(a - b), np.cos(a - b)))))


def validate_success_thresholds(xy_m, yaw_deg):
    if not np.isfinite([xy_m, yaw_deg]).all() or xy_m < 0 or not 0 <= yaw_deg <= 60:
        raise ValueError("Success thresholds must be finite: XY >= 0 m and yaw in [0, 60] degrees")


def success_from_errors(xy, angle, xy_m=SUCCESS_XY_M, yaw_deg=SUCCESS_YAW_DEG):
    validate_success_thresholds(xy_m, yaw_deg)
    xy, angle = np.asarray(xy), np.asarray(angle)
    # Inclusive thresholds; the tiny tolerance only absorbs floating-point roundoff.
    return (np.isfinite(xy) & np.isfinite(angle) & (xy >= 0) & (angle >= 0)
            & (xy <= xy_m + 1e-9) & (angle <= yaw_deg + 1e-9))


def localization_metrics(pose, gt_position, gt_yaw):
    """Output-pose metrics, kept separate from nearest-keyframe offsets."""
    xy = np.array([pose.position.x, pose.position.y])
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    norm = np.linalg.norm(q)
    angle = float(yaw(q / norm)) if np.isfinite(q).all() and np.isfinite(norm) and norm > 1e-12 else np.nan
    xy_error = float(np.linalg.norm(xy - gt_position[:2])) if np.isfinite(xy).all() else np.nan
    return (*xy, np.degrees(angle), xy_error, yaw_error_deg(angle, gt_yaw))


class GroundTruth:
    def __init__(self, poses, max_gap):
        poses = poses[np.argsort(poses[:, 0], kind="stable")]
        duplicate = np.flatnonzero(np.diff(poses[:, 0]) == 0)
        for i in duplicate:
            same_position = np.allclose(poses[i, 1:4], poses[i + 1, 1:4], rtol=0, atol=1e-6)
            same_rotation = abs(np.dot(poses[i, 4:8], poses[i + 1, 4:8])) > 1 - 1e-10
            if not (same_position and same_rotation):
                raise ValueError("Conflicting GT poses at timestamp %.9f" % poses[i, 0])
        self.poses = poses[np.r_[True, np.diff(poses[:, 0]) > 0]]
        if len(self.poses) < 2:
            raise ValueError("GT needs at least two distinct timestamps")
        self.times, self.max_gap = self.poses[:, 0], max_gap

    def interpolate(self, stamp):
        j = int(np.searchsorted(self.times, stamp))
        if j < len(self.times) and self.times[j] == stamp:
            return self.poses[j, 1:4], float(yaw(self.poses[j, 4:8]))
        if j == 0 or j == len(self.times):
            raise ValueError("outside_gt_time_range")
        a, b = self.poses[j - 1], self.poses[j]
        gap = b[0] - a[0]
        if self.max_gap > 0 and gap > self.max_gap:
            raise ValueError("gt_interpolation_gap")
        fraction = (stamp - a[0]) / gap
        position = a[1:4] + fraction * (b[1:4] - a[1:4])
        # Quaternion SLERP along the shorter arc, including the q / -q case.
        q0, q1 = a[4:8], b[4:8].copy()
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1, dot = -q1, -dot
        if dot > 0.9995:
            q = (1 - fraction) * q0 + fraction * q1
        else:
            angle = np.arccos(np.clip(dot, -1, 1))
            q = (np.sin((1 - fraction) * angle) * q0 + np.sin(fraction * angle) * q1) / np.sin(angle)
        return position, float(yaw(q / np.linalg.norm(q)))


def mean_score_by_error(scores, errors, edges):
    """Arithmetic mean of raw scores per error bin; empty bins break the line."""
    counts, _ = np.histogram(errors, bins=edges)
    totals, _ = np.histogram(errors, bins=edges, weights=scores)
    means = np.full(len(counts), np.nan)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means, counts


def validate_plot_filters(min_bin_count, exclude_xy_below_m, exclude_yaw_below_deg):
    if (not np.isfinite([min_bin_count, exclude_xy_below_m, exclude_yaw_below_deg]).all()
            or min_bin_count < 1 or int(min_bin_count) != min_bin_count
            or exclude_xy_below_m < 0 or not 0 <= exclude_yaw_below_deg <= 180):
        raise ValueError("Plot filters require an integer min_bin_count >= 1, XY >= 0 and yaw in [0, 180]")


def plot_sample_mask(samples, exclude_xy_below_m=0.01, exclude_yaw_below_deg=0.5):
    """Exclude only jointly near-zero GT-to-nearest-keyframe offsets, not output-pose errors."""
    values = np.asarray(samples, dtype=float).reshape(-1, 3)
    return ~((values[:, 1] < exclude_xy_below_m) & (values[:, 2] < exclude_yaw_below_deg))


def validate_non_neighbor_table(data, keyframes):
    table = np.asarray(data, dtype=float).reshape(-1, len(NON_NEIGHBOR_FIELDS))
    n = len(keyframes)
    if len(table) != n or not np.array_equal(table[:, 0], np.arange(n)):
        raise ValueError("Non-neighbor table does not match map keyframe indices/count")
    if not np.isfinite(table[:, :8]).all() or not np.allclose(table[:, 1:4], keyframes[:, 1:4], rtol=0, atol=1e-5):
        raise ValueError("Non-neighbor table and map positions differ")
    norms = np.linalg.norm(table[:, 4:8], axis=1)
    if np.any(norms < 1e-12) or not np.allclose(abs(np.sum(table[:, 4:8] / norms[:, None] * keyframes[:, 4:8], axis=1)), 1, rtol=0, atol=1e-7):
        raise ValueError("Non-neighbor table and map orientations differ")
    scores, ids, counts = table[:, 8:11].T
    valid = np.isfinite(scores)
    if (np.isinf(scores).any() or np.any(scores[valid] < 0) or np.any(scores[valid] > 2)
            or not np.isfinite(table[:, 9:]).all() or np.any(ids != np.rint(ids))
            or np.any((ids < -1) | (ids >= n)) or np.any(counts != np.rint(counts))
            or np.any((counts < 0) | (counts >= n)) or not np.array_equal(valid, (ids >= 0) & (counts > 0))
            or np.any(ids[~valid] != -1) or np.any(counts[~valid] != 0)):
        raise ValueError("Invalid non-neighbor scores/indices/comparison counts")
    source, target = np.flatnonzero(valid), ids[valid].astype(int)
    xy = np.linalg.norm(keyframes[source, 1:3] - keyframes[target, 1:3], axis=1)
    delta = yaw(keyframes[source, 4:8]) - yaw(keyframes[target, 4:8])
    angle = np.degrees(abs(np.arctan2(np.sin(delta), np.cos(delta))))
    if np.any(success_from_errors(xy, angle, SUCCESS_XY_M, SUCCESS_YAW_DEG)):
        raise ValueError("Non-neighbor table contains a geometrically valid neighbor")
    return table


def load_non_neighbor_csv(path, keyframes):
    """Read the offline table and validate geometry, indices and threshold metadata."""
    with Path(path).expanduser().open(encoding="utf-8-sig", newline="") as file:
        lines = file.readlines()
    metadata = dict(line.lstrip()[1:].strip().split("=", 1) for line in lines
                    if line.lstrip().startswith("#") and "=" in line)
    reader = csv.DictReader(line for line in lines if line.strip() and not line.lstrip().startswith("#"))
    if not set(NON_NEIGHBOR_FIELDS).issubset(reader.fieldnames or []):
        raise ValueError("Non-neighbor CSV is missing required columns")
    for name, expected in (("xy_threshold", SUCCESS_XY_M), ("yaw_threshold_deg", SUCCESS_YAW_DEG)):
        if name in metadata and not np.isclose(float(metadata[name]), expected, rtol=0, atol=1e-9):
            raise ValueError("Non-neighbor CSV has a different " + name + str(SUCCESS_XY_M))
    table = [[float(row[name]) for name in NON_NEIGHBOR_FIELDS] for row in reader]
    return validate_non_neighbor_table(table, keyframes), metadata


def plot_success_curve(success, errors, edges, title, xlabel, output_dir, name, min_bin_count=100):
    rates, counts = mean_score_by_error(success.astype(float), errors, edges)
    valid = counts >= min_bin_count
    centers = (edges[:-1] + edges[1:]) / 2
    successes = np.rint(np.nan_to_num(rates) * counts).astype(int)
    with (output_dir / (name + ".csv")).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("error_lower", "error_upper", "error_center", "success_rate", "successes", "count", "plotted"))
        writer.writerows(zip(edges[:-1], edges[1:], centers, rates, successes, counts, valid.astype(int)))
    fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
    ax.plot(centers, np.where(valid, 100 * rates, np.nan), "o-", color="#e74324", linewidth=1.8, markersize=3, clip_on=False)
    ax.set(xlabel=xlabel, ylabel="Localization success rate (%)", title=title,
           xlim=(edges[0], edges[-1]), ylim=(0, 100), yticks=np.arange(0, 101, 20))
    ax.grid(alpha=0.25)
    if not valid.any():
        ax.text(0.5, 0.5, "No bins with >= %d samples" % min_bin_count, ha="center", va="center", transform=ax.transAxes)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / (name + "." + extension), dpi=300)
    plt.close(fig)


def plot_non_neighbor_curve(scores, success, edges, title, output_dir, name, missing, count_norm=None, min_bin_count=100):
    rates, counts = mean_score_by_error(success.astype(float), scores, edges)
    valid = counts >= min_bin_count
    centers = (edges[:-1] + edges[1:]) / 2
    successes = np.rint(np.nan_to_num(rates) * counts).astype(int)
    with (output_dir / (name + ".csv")).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("score_lower", "score_upper", "score_center", "success_rate", "successes", "count", "plotted"))
        writer.writerows(zip(edges[:-1], edges[1:], centers, rates, successes, counts, valid.astype(int)))
    fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
    if count_norm is None:
        count_norm = LogNorm(vmin=min_bin_count, vmax=max(min_bin_count + 1, int(counts.max())))
    ax.plot(centers, np.where(valid, 100 * rates, np.nan), color="#777777", linewidth=1.2, zorder=2)
    points = ax.scatter(centers[valid], 100 * rates[valid], c=counts[valid], cmap="viridis", norm=count_norm,
                        s=30, edgecolors="white", linewidths=0.5, zorder=3, clip_on=False)
    fig.colorbar(points, ax=ax, label="Samples per score bin (log scale)")
    ax.set(xlabel="Non-neighbor score (min cosine distance)", ylabel="Localization success rate (%)",
           title=title + "   (N=%d, missing=%d)" % (len(scores), missing),
           xlim=(edges[0], edges[-1]), ylim=(0, 100), yticks=np.arange(0, 101, 20))
    ax.grid(alpha=0.25)
    if not valid.any():
        ax.text(0.5, 0.5, "No bins with >= %d samples" % min_bin_count, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / (name + "." + extension), dpi=300)
    plt.close(fig)


def validate_yaw_groups(low, high):
    if not np.isfinite([low, high]).all() or not 0 < low < high < 60:
        raise ValueError("Yaw group thresholds must satisfy 0 < low < high < 60 degrees")


def plot_histograms(samples, output_dir, score_bins=50, error_bins=50, success=None, non_neighbor=None,
                    yaw_group_low_deg=5.0, yaw_group_high_deg=15.0, min_bin_count=100,
                    exclude_xy_below_m=0.01, exclude_yaw_below_deg=0.5):
    """Rows are (score, XY offset in metres, absolute wrapped yaw offset in degrees)."""
    values = np.asarray(samples, dtype=float).reshape(-1, 3)
    validate_yaw_groups(yaw_group_low_deg, yaw_group_high_deg)
    validate_plot_filters(min_bin_count, exclude_xy_below_m, exclude_yaw_below_deg)
    if min(score_bins, error_bins) < 1:
        raise ValueError("Histogram bin counts must be positive")
    if not np.isfinite(values).all() or np.any(values[:, 1:] < 0) or np.any(values[:, 2] > 60):
        raise ValueError("Samples must be finite, with XY >= 0 and yaw in [0, 60]")
    if success is not None:
        success = np.asarray(success)
        if success.shape != (len(values),) or not np.isin(success, [0, 1]).all():
            raise ValueError("Success must contain one binary result per sample")
    if non_neighbor is not None:
        non_neighbor = np.asarray(non_neighbor, dtype=float)
        if success is None or non_neighbor.shape != (len(values),):
            raise ValueError("Non-neighbor scores require one success result and score per sample")
        known = np.isfinite(non_neighbor)
        if np.isinf(non_neighbor).any() or np.any((non_neighbor[known] < 0) | (non_neighbor[known] > 2)):
            raise ValueError("Non-neighbor scores must be cosine distances in [0, 2] or NaN")
    keep = plot_sample_mask(values, exclude_xy_below_m, exclude_yaw_below_deg)
    values = values[keep]
    if success is not None:
        success = success[keep]
    if non_neighbor is not None:
        non_neighbor = non_neighbor[keep]
        known = np.isfinite(non_neighbor)
        nn_edges = np.linspace(0, max(0.01, float(non_neighbor[known].max())) if known.any() else 1, score_bins + 1)
    score, xy, angle = values.T
    lo, hi = (min(0.0, float(score.min())), max(0.01, float(score.max()))) if len(values) else (0.0, 1.0)
    score_edges = np.linspace(lo, hi, score_bins + 1)
    xy_edges = np.linspace(0, max(1.0, float(xy.max())) if len(values) else 1.0, error_bins + 1)
    yaw_edges = np.linspace(0, 60, error_bins + 1)
    groups = [angle < yaw_group_low_deg, (angle >= yaw_group_low_deg) & (angle <= yaw_group_high_deg), angle > yaw_group_high_deg,
              xy < 1, (xy >= 1) & (xy < 3), xy >= 3]
    if non_neighbor is not None:
        max_count = max(np.histogram(non_neighbor[mask & known], bins=nn_edges)[0].max() for mask in groups)
        nn_count_norm = LogNorm(vmin=min_bin_count, vmax=max(min_bin_count + 1, int(max_count)))
    low, high = format(yaw_group_low_deg, "g"), format(yaw_group_high_deg, "g")
    names = [f"01_score_xy_yaw_lt{low}", f"02_score_xy_yaw_{low}to{high}", f"03_score_xy_yaw_gt{high}",
             "04_score_yaw_xy_lt1", "05_score_yaw_xy_1to3", "06_score_yaw_xy_ge3"]
    titles = [rf"$0\leq e_\psi<{low}^\circ$", rf"${low}^\circ\leq e_\psi\leq{high}^\circ$", rf"$e_\psi>{high}^\circ$",
              r"$0\leq e_{xy}<1$ m", r"$1\leq e_{xy}<3$ m", r"$e_{xy}\geq3$ m"]
    histograms = []
    for i, mask in enumerate(groups):
        edges, errors = (xy_edges, xy) if i < 3 else (yaw_edges, angle)
        counts, _, _ = np.histogram2d(score[mask], errors[mask], bins=(score_edges, edges))
        histograms.append(counts)
    # Filter each 2-D cell separately; curves use their own 1-D bin counts.
    norm = LogNorm(vmin=min_bin_count, vmax=max(min_bin_count + 1, max(h.max() for h in histograms)))
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad("white")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 12, "svg.fonttype": "none"}):
        for i, (mask, counts) in enumerate(zip(groups, histograms)):
            edges = xy_edges if i < 3 else yaw_edges
            errors = xy if i < 3 else angle
            centers = (edges[:-1] + edges[1:]) / 2
            means, bin_counts = mean_score_by_error(score[mask], errors[mask], edges)
            valid = bin_counts >= min_bin_count
            with (output_dir / (names[i] + "_mean.csv")).open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("error_lower", "error_upper", "error_center", "mean_score", "count", "plotted"))
                writer.writerows(zip(edges[:-1], edges[1:], centers, means, bin_counts, valid.astype(int)))
            fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
            mesh = ax.pcolormesh(edges, score_edges, np.ma.masked_less(counts, min_bin_count), cmap=cmap, norm=norm)
            xlabel = "XY offset to nearest keyframe (m)" if i < 3 else "Yaw offset to nearest keyframe (deg)"
            if valid.any():
                ax.plot(centers, np.where(valid, means, np.nan), "o-", color="#e74324", linewidth=1.8, markersize=3,
                        markeredgecolor="white", markeredgewidth=0.5, label="Mean score", zorder=3,
                        path_effects=[path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()])
                ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
            ax.set(xlabel=xlabel,
                   ylabel="Score (cosine distance, lower is better)",
                   title=titles[i] + "   (N=%d)" % mask.sum())
            if not valid.any():
                ax.text(0.5, 0.5, "No bins with >= %d samples" % min_bin_count, ha="center", va="center", transform=ax.transAxes)
            fig.colorbar(mesh, ax=ax, label="Frame count (log scale)")
            for extension in ("png", "svg"):
                fig.savefig(output_dir / (names[i] + "." + extension), dpi=300)
            plt.close(fig)
            if success is not None:
                name = "%02d_success_rate_%s" % (i + 7, names[i].split("_score_", 1)[1])
                plot_success_curve(success[mask], errors[mask], edges,
                                   titles[i] + "   (N=%d)" % mask.sum(), xlabel, output_dir, name, min_bin_count)
            if non_neighbor is not None:
                eligible = mask & known
                name = "%02d_non_neighbor_success_%s" % (i + 13, names[i].split("_score_", 1)[1])
                plot_non_neighbor_curve(non_neighbor[eligible], success[eligible], nn_edges, titles[i], output_dir,
                                        name, int((mask & ~known).sum()), nn_count_norm, min_bin_count)
    np.savez_compressed(output_dir / "histogram_counts.npz", score_edges=score_edges,
                        xy_edges=xy_edges, yaw_edges=yaw_edges, **dict(zip(names, histograms)))
    return {name: int(mask.sum()) for name, mask in zip(names, groups)}


def replot_csv(csv_path, output_dir=None, score_bins=None, error_bins=None,
               yaw_group_low_deg=None, yaw_group_high_deg=None, success_xy_m=None, success_yaw_deg=None,
               min_bin_count=None, exclude_xy_below_m=None, exclude_yaw_below_deg=None):
    """Replot an existing run without ROS, GT files, or replaying the bag."""
    csv_path = Path(csv_path).expanduser()
    summary_path = csv_path.parent / "summary.json"
    config = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    success_xy_m = float(config.get("/success_xy_m", SUCCESS_XY_M)) if success_xy_m is None else success_xy_m
    success_yaw_deg = float(config.get("/success_yaw_deg", SUCCESS_YAW_DEG)) if success_yaw_deg is None else success_yaw_deg
    validate_success_thresholds(success_xy_m, success_yaw_deg)
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = ("score", "xy_error_m", "yaw_error_deg")
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain score, xy_error_m and yaw_error_deg")
        rows = list(reader)
        samples = [tuple(float(row[key]) for key in fields) for row in rows]
        error_fields = ("localization_xy_error_m", "localization_yaw_error_deg")
        success = None
        if set(error_fields).issubset(reader.fieldnames):
            success = [bool(success_from_errors(*(float(row[key]) for key in error_fields), success_xy_m, success_yaw_deg))
                       for row in rows]
        else:
            warnings.warn("Old CSV has no output-pose errors; only score plots can be regenerated. "
                          "Replay /localization_odom with the updated node to record success rates.", RuntimeWarning)
        non_neighbor = ([float(row["non_neighbor_score"]) for row in rows]
                        if success is not None and "non_neighbor_score" in reader.fieldnames else None)
    if config.get("non_neighbor_csv") == "" and non_neighbor is not None and not np.isfinite(non_neighbor).any():
        non_neighbor = None
    score_bins = int(config.get("score_bins", 50)) if score_bins is None else score_bins
    error_bins = int(config.get("error_bins", 50)) if error_bins is None else error_bins
    yaw_group_low_deg = float(config.get("yaw_group_low_deg", 5)) if yaw_group_low_deg is None else yaw_group_low_deg
    yaw_group_high_deg = float(config.get("yaw_group_high_deg", 15)) if yaw_group_high_deg is None else yaw_group_high_deg
    min_bin_count = int(config.get("min_bin_count", 100)) if min_bin_count is None else min_bin_count
    exclude_xy_below_m = float(config.get("exclude_xy_below_m", 0.01)) if exclude_xy_below_m is None else exclude_xy_below_m
    exclude_yaw_below_deg = float(config.get("exclude_yaw_below_deg", 0.5)) if exclude_yaw_below_deg is None else exclude_yaw_below_deg
    output_dir = Path(output_dir).expanduser() if output_dir else csv_path.parent / "with_mean"
    plot_histograms(samples, output_dir, score_bins, error_bins, success, non_neighbor, yaw_group_low_deg, yaw_group_high_deg,
                    min_bin_count, exclude_xy_below_m, exclude_yaw_below_deg)
    kept = int(plot_sample_mask(samples, exclude_xy_below_m, exclude_yaw_below_deg).sum())
    settings = dict(source_csv=str(csv_path.resolve()), score_bins=score_bins, error_bins=error_bins,
                    yaw_group_low_deg=yaw_group_low_deg, yaw_group_high_deg=yaw_group_high_deg,
                    min_bin_count=min_bin_count, exclude_xy_below_m=exclude_xy_below_m,
                    exclude_yaw_below_deg=exclude_yaw_below_deg, plot_sample_count=kept,
                    plot_excluded_near_zero=len(samples) - kept,
                    success_xy_m=success_xy_m, success_yaw_deg=success_yaw_deg,
                    success_count=sum(success) if success is not None else None)
    (output_dir / "plot_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return output_dir


class LocalizationHistograms:
    FIELDS = ("stamp", "gt_stamp", "score", "matched_index", "nearest_index", "gt_x", "gt_y", "gt_z",
              "gt_yaw_deg", "keyframe_stamp", "keyframe_x", "keyframe_y", "keyframe_yaw_deg",
              "xy_error_m", "yaw_error_deg", "localization_x", "localization_y", "localization_yaw_deg",
              "localization_xy_error_m", "localization_yaw_error_deg", "success", "non_neighbor_score", "non_neighbor_index")

    def __init__(self):
        import rospy
        from nav_msgs.msg import Odometry
        self.ros = rospy
        param = rospy.get_param
        keyframe_path = param("~keyframe_poses_file", "") or param("/poses_file", "")
        if not keyframe_path:
            raise ValueError("Set ~keyframe_poses_file to the map pose file used by place_recognition")
        self.keyframe_path = os.path.abspath(os.path.expanduser(keyframe_path))
        gt_file = param("~gt_file", "")
        if not gt_file:
            raise ValueError("Set ~gt_file to a trajectory in the map coordinate frame")
        self.gt_path = os.path.abspath(os.path.expanduser(gt_file))
        self.topic = param("~localization_topic", "/localization_odom")
        self.non_neighbor_csv = os.path.expanduser(param("~non_neighbor_csv", ""))
        self.producer_threshold = param("/sc_dist_thres", None)
        self.time_offset = float(param("~gt_time_offset", 0.0))
        max_gap = float(param("~max_gt_gap", 1.0))
        self.score_bins = int(param("~score_bins", 50))
        self.error_bins = int(param("~error_bins", 50))
        self.min_bin_count = int(param("~min_bin_count", 100))
        self.exclude_xy_below_m = float(param("~exclude_xy_below_m", 0.01))
        self.exclude_yaw_below_deg = float(param("~exclude_yaw_below_deg", 0.5))
        validate_plot_filters(self.min_bin_count, self.exclude_xy_below_m, self.exclude_yaw_below_deg)
        self.yaw_group_low_deg = float(param("~yaw_group_low_deg", 5.0))
        self.yaw_group_high_deg = float(param("~yaw_group_high_deg", 15.0))
        validate_yaw_groups(self.yaw_group_low_deg, self.yaw_group_high_deg)
        self.success_xy_m = float(param("/success_xy_m", SUCCESS_XY_M))
        self.success_yaw_deg = float(param("/success_yaw_deg", SUCCESS_YAW_DEG))
        validate_success_thresholds(self.success_xy_m, self.success_yaw_deg)
        if not np.isfinite([self.time_offset, max_gap]).all() or max_gap < 0:
            raise ValueError("gt_time_offset must be finite; max_gt_gap must be finite and >= 0")
        if min(self.score_bins, self.error_bins) < 1:
            raise ValueError("Histogram bin counts must be positive")
        self.keyframes = load_poses(self.keyframe_path)
        self.non_neighbor_table, self.non_neighbor_metadata = None, {}
        if self.non_neighbor_csv:
            self.non_neighbor_csv = os.path.abspath(self.non_neighbor_csv)
            self.non_neighbor_table, self.non_neighbor_metadata = load_non_neighbor_csv(self.non_neighbor_csv, self.keyframes)
        self.tree = cKDTree(self.keyframes[:, 1:3])
        self.keyframe_yaws = yaw(self.keyframes[:, 4:8])
        self.gt = GroundTruth(load_poses(self.gt_path), max_gap)
        root = Path(param("~output_dir", "~/.ros/localization_error_histograms")).expanduser()
        self.output_dir = root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_%d" % os.getpid())
        self.output_dir.mkdir(parents=True)
        if self.non_neighbor_table is not None:
            shutil.copyfile(self.non_neighbor_csv, self.output_dir / "non_neighbor_keyframes.csv")
        self.csv_file = (self.output_dir / "observations.csv").open("w", newline="", buffering=1)
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(self.FIELDS)
        self.samples, self.skipped = [], Counter()
        self.success, self.invalid_localization_poses = [], 0
        self.non_neighbor_scores = []
        self.received, self.closed = 0, False
        self.lock = threading.RLock()
        rospy.on_shutdown(self.save)
        self.sub = rospy.Subscriber(self.topic, Odometry, self.callback, queue_size=1000)
        rospy.loginfo("Histogram monitor: %d map poses; GT [%.9f, %.9f]; output: %s",
                      len(self.keyframes), self.gt.times[0], self.gt.times[-1], self.output_dir)
        rospy.loginfo("Errors = interpolated GT vs XY-nearest MAP keyframe; matched index is logged only.")
        rospy.loginfo("Plot filters: count >= %d per bin; exclude offsets with XY < %g m AND yaw < %g deg. Raw CSV retained.",
                      self.min_bin_count, self.exclude_xy_below_m, self.exclude_yaw_below_deg)
        rospy.loginfo("Yaw plot groups (deg): < %g, [%g, %g], > %g", self.yaw_group_low_deg,
                      self.yaw_group_low_deg, self.yaw_group_high_deg, self.yaw_group_high_deg)
        rospy.loginfo("Success = output pose vs GT: XY <= %g m AND yaw <= %g deg.", self.success_xy_m, self.success_yaw_deg)
        rospy.logwarn("Only published localization results are recorded (producer threshold + registration gating).")
        if self.non_neighbor_table is not None:
            rospy.loginfo("Loaded offline non-neighbor CSV: %s (%d/%d usable keyframes)", self.non_neighbor_csv,
                          np.isfinite(self.non_neighbor_table[:, 8]).sum(), len(self.keyframes))

    def callback(self, msg):
        with self.lock:
            if self.closed:
                return
            self.received += 1
            stamp = msg.header.stamp.to_sec()
            score, raw_index = msg.pose.covariance[:2]
            reason = ""
            if not np.isfinite([stamp, score, raw_index]).all():
                reason = "nonfinite_metadata"
            elif abs(raw_index - round(raw_index)) > 1e-3 or not 0 <= round(raw_index) < len(self.keyframes):
                reason = "invalid_matched_index"
            if not reason:
                try:
                    position, gt_yaw = self.gt.interpolate(stamp + self.time_offset)
                except ValueError as error:
                    reason = str(error)
            if reason:
                self.skipped[reason] += 1
                self.ros.logwarn_throttle(5, "Histogram sample skipped: %s; stamp=%.9f" % (reason, stamp))
                return
            xy_error, nearest = self.tree.query(position[:2])
            nearest = int(nearest)
            keyframe, keyframe_yaw = self.keyframes[nearest], self.keyframe_yaws[nearest]
            yaw_error = yaw_error_deg(gt_yaw, keyframe_yaw)
            metrics = localization_metrics(msg.pose.pose, position, gt_yaw)
            success = bool(success_from_errors(*metrics[-2:], self.success_xy_m, self.success_yaw_deg))
            if not np.isfinite(metrics[-2:]).all():
                self.invalid_localization_poses += 1
                self.ros.logwarn_throttle(5, "Invalid output pose counted as a localization failure.")
            nn_score, nn_index = self.non_neighbor_table[nearest, 8:10] if self.non_neighbor_table is not None else (np.nan, -1)
            self.writer.writerow(("%.9f" % stamp, "%.9f" % (stamp + self.time_offset), score,
                                  int(round(raw_index)), nearest, *position, np.degrees(gt_yaw), keyframe[0],
                                  *keyframe[1:3], np.degrees(keyframe_yaw), xy_error, yaw_error, *metrics, int(success), nn_score, int(nn_index)))
            self.samples.append((score, xy_error, yaw_error))
            self.success.append(success)
            self.non_neighbor_scores.append(nn_score)
            self.ros.loginfo_throttle(10, "Histogram monitor: received=%d, valid=%d, skipped=%d" %
                                     (self.received, len(self.samples), sum(self.skipped.values())))

    def save(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.csv_file.close()
            samples = self.samples.copy()
            success = self.success.copy()
            non_neighbor = np.asarray(self.non_neighbor_scores) if self.non_neighbor_table is not None else None
            nn_valid = int(np.isfinite(self.non_neighbor_scores).sum())
            kept = int(plot_sample_mask(samples, self.exclude_xy_below_m, self.exclude_yaw_below_deg).sum())
            summary = dict(received=self.received, valid=len(samples), skipped=dict(self.skipped),
                           keyframe_poses_file=self.keyframe_path, gt_file=self.gt_path,
                           localization_topic=self.topic, gt_time_offset=self.time_offset,
                           max_gt_gap=self.gt.max_gap, score_bins=self.score_bins, error_bins=self.error_bins,
                           min_bin_count=self.min_bin_count, exclude_xy_below_m=self.exclude_xy_below_m,
                           exclude_yaw_below_deg=self.exclude_yaw_below_deg, plot_sample_count=kept,
                           plot_excluded_near_zero=len(samples) - kept,
                           yaw_group_low_deg=self.yaw_group_low_deg, yaw_group_high_deg=self.yaw_group_high_deg,
                           error_reference="interpolated GT vs XY-nearest map keyframe",
                           angle_definition="absolute shortest yaw difference in [0, 60] degrees",
                           success_definition="output pose vs interpolated GT: XY <= %g m AND yaw <= %g deg" %
                                              (self.success_xy_m, self.success_yaw_deg),
                           success_xy_m=self.success_xy_m, success_yaw_deg=self.success_yaw_deg,
                           success_count=sum(success), success_rate=float(np.mean(success)) if success else None,
                           invalid_localization_poses=self.invalid_localization_poses,
                           success_denominator="received samples with valid metadata and interpolated GT",
                           non_neighbor_csv=self.non_neighbor_csv, non_neighbor_metadata=self.non_neighbor_metadata,
                           non_neighbor_definition="min cosine distance to map frames with XY > 3 m OR yaw > 15 deg",
                           non_neighbor_valid_samples=nn_valid,
                           non_neighbor_missing_samples=len(samples) - nn_valid,
                           producer_sc_dist_thres_at_start=self.producer_threshold)
        try:
            summary["groups"] = plot_histograms(samples, self.output_dir, self.score_bins, self.error_bins, success, non_neighbor,
                                               self.yaw_group_low_deg, self.yaw_group_high_deg, self.min_bin_count,
                                               self.exclude_xy_below_m, self.exclude_yaw_below_deg)
            self.ros.loginfo("Saved %d plots (%d samples after near-zero exclusion, %d raw) to %s",
                             18 if non_neighbor is not None else 12, kept, len(samples), self.output_dir)
        except Exception as error:
            summary["plot_error"] = str(error)
            self.ros.logerr("Plotting failed: %s. Observations are saved in %s", error, self.output_dir)
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor ROS localization, or replot score/non-neighbor/success curves from CSV.")
    parser.add_argument("--plot-csv", help="Path to an existing observations.csv; no ROS required")
    parser.add_argument("--output-dir", help="Replot directory (default: CSV directory / with_mean)")
    parser.add_argument("--score-bins", type=int, help="Replot score bins (default: saved settings, or 50)")
    parser.add_argument("--error-bins", type=int, help="Replot error/mean bins (default: saved settings, or 50)")
    parser.add_argument("--min-bin-count", type=int, help="Minimum samples per plotted bin (default: saved setting, or 100)")
    parser.add_argument("--exclude-xy-below-m", type=float, help="Joint near-zero exclusion: XY threshold in metres (default: 0.01)")
    parser.add_argument("--exclude-yaw-below-deg", type=float, help="Joint near-zero exclusion: yaw threshold in degrees (default: 0.5)")
    parser.add_argument("--yaw-group-low-deg", type=float, help="Lower yaw group threshold (default: saved setting, or 5)")
    parser.add_argument("--yaw-group-high-deg", type=float, help="Upper yaw group threshold, included in middle group (default: saved setting, or 15)")
    parser.add_argument("--success-xy-m", type=float, help="Success XY error threshold in metres (default: saved setting, or 3)")
    parser.add_argument("--success-yaw-deg", type=float, help="Success yaw error threshold in degrees (default: saved setting, or 15)")
    args, ros_args = parser.parse_known_args()
    if args.plot_csv:
        if ros_args:
            parser.error("Unexpected arguments: " + " ".join(ros_args))
        try:
            output = replot_csv(args.plot_csv, args.output_dir, args.score_bins, args.error_bins,
                                args.yaw_group_low_deg, args.yaw_group_high_deg, args.success_xy_m, args.success_yaw_deg,
                                args.min_bin_count, args.exclude_xy_below_m, args.exclude_yaw_below_deg)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
        print("Saved score plots and available success-rate curves to %s" % output)
        raise SystemExit(0)
    import rospy
    rospy.init_node("localization_error_histograms")
    try:
        node = LocalizationHistograms()
    except (OSError, ValueError) as error:
        rospy.logfatal("Cannot start histogram monitor: %s", error)
        raise SystemExit(1)
    rospy.spin()
