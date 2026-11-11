#!/usr/bin/env python3
"""Compare recorded algorithms offline; automatically read ROS parameters under roslaunch.

Score handling
--------------
score_mode=native (default) sweeps each algorithm's own unique scores.  This is
required when RaPlace (~1e-3), RC/SC (~1e-1) and ReFeree (~1e2) are compared:
a shared grid either loses resolution or produces too many thresholds.  The
accepted-fraction vs recall figure is invariant to score scale and is the
primary output; score_normalization only changes how the score-threshold panels
are displayed (minmax, percentile or none).
"""
import argparse
import csv
import json
import textwrap
import warnings
from decimal import Decimal
from colorsys import hsv_to_rgb
from pathlib import Path

import numpy as np


DEFAULTS = dict(csv_dir="~/.ros/localization_recall", csv_files=[], algorithm_count=3,
                algorithm_labels=[], algorithm_score_maxima=[], output_dir="", recall_thresholds=[2., 5., 10.],
                score_min=0., score_max=1., score_step=0.02, score_mode="native",
                score_normalization="minmax", error_source="localization",
                distance_mode="xy", denominator="accepted", total_queries=0,
                paper_width_mm=180., paper_height_mm=58., font_size_pt=9., legend_font_size_pt=9.)


def score_grid(low, high, step):
    if not np.isfinite([low, high, step]).all() or not 0 <= low < high or step <= 0:
        raise ValueError("Require 0 <= score_min < score_max and score_step > 0")
    a, b, s = map(lambda v: Decimal(str(v)), (low, high, step))
    count = int((b - a) // s) + 1
    if count > 10000:
        raise ValueError("Too many score thresholds (maximum 10000)")
    values = [a + i * s for i in range(count)]
    return np.array([float(v) for v in values])


def native_thresholds(scores):
    """Every reachable acceptance point of one algorithm, regardless of score scale.

    A strict '<' gate just above each observed score accepts that score's whole tie
    group.  This avoids a shared score grid, which cannot cover RaPlace (~1e-3) and
    ReFeree (~1e2) at the same time without either losing resolution or exploding
    the number of thresholds.
    """
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    return np.nextafter(np.unique(values), np.inf) if values.size else np.array([], dtype=float)


def set_score_statistics(run):
    """Cache the observed score range and sorted scores for display normalization."""
    values = np.asarray(run["all_scores"], dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    run["score_lo"] = float(values.min()) if values.size else 0.0
    run["score_hi"] = float(values.max()) if values.size else 0.0
    run["score_span"] = run["score_hi"] - run["score_lo"]
    run["score_sorted"] = np.sort(values)


def attach_score_axes(rows, runs):
    """Add a per-algorithm normalized score and percentile to every curve row.

    The acceptance/recall curves are still computed from each algorithm's own native
    thresholds; these columns only reparameterize the score axis for display.
    """
    by_label = {run["label"]: run for run in runs}
    for row in rows:
        run = by_label[row["algorithm"]]
        threshold = float(row["sc_dist_thres"])
        span = run["score_span"]
        row["score_normalized"] = float(np.clip((threshold - run["score_lo"]) / span, 0.0, 1.0)) if span > 0 else 0.0
        n = len(run["score_sorted"])
        row["score_percentile_pct"] = float(
            100.0 * np.searchsorted(run["score_sorted"], threshold, side="left") / n) if n else 0.0


def load_run(path, error_column, label=""):
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    records, invalid_gt, seen, all_scores = {}, set(), set(), []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"stamp_ns", "score", "gt_valid", error_column}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing columns in %s: %s" % (path, required - set(reader.fieldnames or [])))
        for row in reader:
            stamp = int(row["stamp_ns"])
            if stamp <= 0 or stamp in seen:
                raise ValueError("Invalid/duplicate timestamp %s in %s" % (stamp, path))
            seen.add(stamp)
            if row["gt_valid"] not in ("0", "1"):
                raise ValueError("gt_valid must be 0 or 1 in " + str(path))
            score = float(row["score"])
            all_scores.append(score)
            if row["gt_valid"] == "0":
                invalid_gt.add(stamp)
            else:
                records[stamp] = (score, float(row[error_column]))
    return dict(path=str(path), label=label or metadata.get("algorithm_name") or path.stem,
                metadata=metadata, records=records, invalid_gt=invalid_gt, all_scores=all_scores)


def validate_runs(runs, score_max, error_source, score_mode="grid"):
    fields = ["gt_sha256", "gt_time_offset", "max_gt_gap"]
    if error_source == "matched":
        fields.append("map_sha256")
    for field in fields:
        values = {run["metadata"][field] for run in runs if field in run["metadata"]}
        if len(values) > 1:
            raise ValueError("Runs have inconsistent %s; use the same evaluation data/settings" % field)
    for run in runs:
        gate = run["metadata"].get("producer_sc_dist_thres")
        maximum = run.get("score_max", score_max)
        if gate is None:
            warnings.warn("%s: recording-time sc_dist_thres is unknown; confirm it covered the observed scores" % run["label"])
        elif not np.isfinite(float(gate)) or float(gate) < 0:
            raise ValueError("%s: invalid recorded sc_dist_thres=%s" % (run["label"], gate))
        elif float(gate) < maximum:
            message = ("%s: recorded sc_dist_thres=%s is below the observed maximum %s; "
                       "high-score outputs may have been censored" % (run["label"], gate, maximum))
            if score_mode == "native":
                warnings.warn(message)
            else:
                raise ValueError(message + " (use score_mode:=native for native thresholds)")
        if run["metadata"].get("status") == "recording":
            warnings.warn("%s: recorder did not finish cleanly; CSV may be incomplete" % run["label"])


def evaluate(runs, thresholds, distances, denominator="accepted", total_queries=0):
    queries = set().union(*(run["records"] for run in runs))
    if not queries:
        raise ValueError("No observations with valid interpolated GT")
    if any(queries & run["invalid_gt"] for run in runs):
        raise ValueError("GT validity differs at shared timestamps; check GT/interpolation settings")
    if total_queries < 0 or (total_queries and total_queries < len(queries)):
        raise ValueError("total_queries must be 0 (auto) or at least the union query count (%d)" % len(queries))
    if denominator not in ("fixed", "accepted"):
        raise ValueError("denominator must be fixed or accepted")
    if denominator == "accepted" and total_queries:
        raise ValueError("total_queries applies only to denominator=fixed")
    total = total_queries or len(queries)
    curves = []
    for run in runs:
        samples = np.array(list(run["records"].values()), dtype=float).reshape(-1, 2)
        scores, errors = samples[:, 0], samples[:, 1]
        valid_score = np.isfinite(scores) & (scores >= 0)
        valid_error = np.isfinite(errors) & (errors >= 0)
        for distance in distances:
            for threshold in run.get("score_thresholds", thresholds):
                accepted = valid_score & (scores < threshold)  # Same strict gate as Scancontext.cpp.
                success = accepted & valid_error & (errors < distance)
                n_accepted, n_success = int(accepted.sum()), int(success.sum())
                n = total if denominator == "fixed" else n_accepted
                curves.append(dict(algorithm=run["label"], recall_distance_m=float(distance),
                    sc_dist_thres=float(threshold), success_count=n_success, accepted_count=n_accepted,
                    observed_queries=len(samples), union_queries=len(queries), total_queries=total,
                    denominator_count=n, success_rate_pct=100. * n_success / n if n else np.nan))
    return curves, queries, total


def evaluate_acceptance(runs, thresholds):
    """Score acceptance uses every recorded output, including rows without GT."""
    curves = []
    for run in runs:
        scores = np.asarray(run["all_scores"], dtype=float)
        valid = np.isfinite(scores) & (scores >= 0)
        for threshold in run.get("score_thresholds", thresholds):
            accepted = int(np.count_nonzero(valid & (scores < threshold)))
            curves.append(dict(algorithm=run["label"], sc_dist_thres=float(threshold),
                score_accepted_count=accepted, recorded_count=len(scores),
                accepted_fraction_pct=100. * accepted / len(scores) if len(scores) else np.nan))
    return curves


def pair_acceptance_recall(curves, acceptance):
    """Join the existing metrics at the same algorithm and score threshold."""
    lookup = {(row["algorithm"], row["sc_dist_thres"]): row for row in acceptance}
    paired = []
    for row in curves:
        accepted = lookup[(row["algorithm"], row["sc_dist_thres"])]
        paired.append(dict(row, accepted_fraction_pct=accepted["accepted_fraction_pct"],
                           score_accepted_count=accepted["score_accepted_count"],
                           recorded_count=accepted["recorded_count"]))
    return paired


def line_style(index):
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p"]
    lines = ["-", "--", "-.", ":"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6F4E7C", "#795548"]
    color = colors[index] if index < len(colors) else hsv_to_rgb((index * .61803398875) % 1, .65, .75)
    # Additional algorithms retain a unique (marker, line) pair.
    return dict(color=color, marker=markers[index % len(markers)],
                linestyle=lines[(index + index // len(markers)) % len(lines)],
                markerfacecolor="white", markeredgecolor=color, markersize=2.8,
                markeredgewidth=0.65, linewidth=1.0)


def compare(config):
    config = dict(DEFAULTS, **config)
    for name in ("csv_files", "algorithm_labels", "algorithm_score_maxima", "recall_thresholds"):
        if not isinstance(config[name], (list, tuple)):
            raise ValueError(name + " must be a list (use YAML lists in launch arguments)")
    if config["error_source"] not in ("matched", "localization") or config["distance_mode"] not in ("xy", "xyz"):
        raise ValueError("error_source=matched/localization; distance_mode=xy/xyz")
    if config["score_mode"] not in ("native", "grid"):
        raise ValueError("score_mode must be native or grid")
    if config["score_normalization"] not in ("minmax", "percentile", "none"):
        raise ValueError("score_normalization must be minmax, percentile or none")
    if config["algorithm_count"] < 1:
        raise ValueError("algorithm_count must be positive")
    for name in ("paper_width_mm", "paper_height_mm", "font_size_pt", "legend_font_size_pt"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name + " must be finite and positive")
    directory = Path(config["csv_dir"]).expanduser().resolve()
    files = config["csv_files"] or ["alg%d.csv" % i for i in range(1, config["algorithm_count"] + 1)]
    paths = [(directory / Path(name).expanduser()).resolve() for name in files]
    labels = config["algorithm_labels"] or [""] * len(paths)
    if len(labels) != len(paths) or len(paths) > 48 or len(set(paths)) != len(paths):
        raise ValueError("Use 1-48 distinct CSV files and one label per algorithm")
    distances = np.array(config["recall_thresholds"], dtype=float)
    if (distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all()
            or np.any(distances <= 0) or len(set(distances)) != len(distances)):
        raise ValueError("recall_thresholds must contain distinct positive distances")
    error_column = config["error_source"] + "_" + config["distance_mode"] + "_error_m"
    runs = [load_run(path, error_column, label) for path, label in zip(paths, labels)]
    if len({run["label"] for run in runs}) != len(runs):
        raise ValueError("Algorithm labels must be distinct; set algorithm_labels explicitly")
    for run in runs:
        set_score_statistics(run)

    if config["score_mode"] == "native":
        if config["algorithm_score_maxima"]:
            warnings.warn("algorithm_score_maxima is ignored when score_mode=native")
        thresholds = None
        for run in runs:
            run["score_thresholds"] = native_thresholds(run["all_scores"])
            run["score_max"] = run["score_hi"]
            if not len(run["score_thresholds"]):
                warnings.warn("%s: no finite nonnegative scores; no curve will be produced" % run["label"])
    else:
        thresholds = score_grid(config["score_min"], config["score_max"], config["score_step"])
        maxima = config["algorithm_score_maxima"]
        if maxima and len(maxima) != len(runs):
            raise ValueError("algorithm_score_maxima must be empty or have one value per algorithm")
        for run, maximum in zip(runs, maxima or [config["score_max"]] * len(runs)):
            run["score_thresholds"] = score_grid(config["score_min"], maximum, config["score_step"])
            run["score_max"] = float(maximum)
        for run in runs:
            count = len(run["score_sorted"])
            above = int(np.count_nonzero(run["score_sorted"] > run["score_max"]))
            if count and above == count:
                raise ValueError(
                    "%s: all %d scores exceed score_max=%g; this algorithm can never be accepted. "
                    "Use score_mode:=native (or set a per-algorithm score_max that covers the observed range)."
                    % (run["label"], count, run["score_max"]))
            if above:
                warnings.warn("%s: %d/%d scores exceed score_max=%g; they can never be accepted"
                              % (run["label"], above, count, run["score_max"]))

    validate_runs(runs, config["score_max"], config["error_source"], config["score_mode"])
    curves, queries, total = evaluate(runs, thresholds, distances, config["denominator"], config["total_queries"])
    acceptance = evaluate_acceptance(runs, thresholds)
    attach_score_axes(curves, runs)
    attach_score_axes(acceptance, runs)
    paired = pair_acceptance_recall(curves, acceptance)
    if len(runs) > 1 and not set.intersection(*(set(run["records"]) for run in runs)):
        warnings.warn("No timestamp appears in every run. Confirm all algorithms used the same query bag and original stamps.")
    if config["denominator"] == "fixed" and not config["total_queries"]:
        warnings.warn("Fixed denominator uses the union of published, GT-valid timestamps. Queries missing from ALL runs are unknown; set total_queries if known.")
    output = Path(config["output_dir"]).expanduser().resolve() if config["output_dir"] else directory / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("recall_curves.csv", curves), ("acceptance_curves.csv", acceptance),
                       ("recall_vs_acceptance_curves.csv", paired)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(config=config, error_column=error_column, union_queries=len(queries), total_queries=total,
        criterion="score < sc_dist_thres AND distance < recall_distance_m; no yaw criterion",
        denominator_note="Fixed: common query union or explicit total_queries; accepted: score-passing results per algorithm",
        acceptance_note="Finite nonnegative score < threshold / all rows in each CSV, including rows without GT; independent of recall distance",
        recall_vs_acceptance_note="X=accepted_fraction_pct, Y=success_rate_pct, paired by algorithm and sc_dist_thres; original denominators unchanged. One subplot per recall distance.",
        score_note="native: each algorithm sweeps its own unique scores (works for ~1e-3 to ~1e3); score_normalized and score_percentile_pct are display-only. grid: shared score_grid from score_min/score_max/score_step.",
        runs=[dict(csv=run["path"], label=run["label"], recorded_results=len(run["all_scores"]), gt_valid_results=len(run["records"]),
                   excluded_no_gt=len(run["invalid_gt"]), missing_from_union=len(queries - run["records"].keys()),
                   metadata=run["metadata"]) for run in runs])
    (output / "comparison_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    draw_curves(curves, acceptance, runs, distances, thresholds, config, output, paired)
    print("Saved %d individual plots + comparison_row + recall_vs_acceptance (PNG + PDF) and curve counts to %s; error=%s; denominator=%s" %
          (1 + len(distances), output, error_column, config["denominator"]))
    return report


def draw_curves(curves, acceptance, runs, distances, thresholds, config, output, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    metric = "Recall" if config["denominator"] == "fixed" else "Localization success"
    plots = [("acceptance", acceptance, "accepted_fraction_pct", "Accepted", "Accepted results", None)]
    for distance in distances:
        name = "recall_%sm" % format(float(distance), ".12g").replace(".", "p")
        plots.append((name, curves, "success_rate_pct", metric, "Recall@%g m" % distance, distance))
    width, height = config["paper_width_mm"] / 25.4, config["paper_height_mm"] / 25.4
    font, legend_font = config["font_size_pt"], config["legend_font_size_pt"]

    def panel(ax, spec, title, acceptance_x=False):
        _, rows, field, ylabel, _, distance = spec
        if acceptance_x:
            xlabel, xlim = "Effective output rate (%)", (0, 100)
            targets = np.array([0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        elif config["score_normalization"] == "minmax":
            xlabel, xlim = "Normalized score", (0, 1)
            targets = np.linspace(0, 1, 11)
        elif config["score_normalization"] == "percentile":
            xlabel, xlim = "Score percentile (%)", (0, 100)
            targets = np.linspace(0, 100, 11)
        else:
            xlabel = "Score threshold"
            xlim = (config["score_min"], max(run.get("score_max", config["score_max"]) for run in runs))
            targets = np.linspace(xlim[0], xlim[1], 11)
        for i, run in enumerate(runs):
            selected = [row for row in rows if row["algorithm"] == run["label"]
                        and (distance is None or row["recall_distance_m"] == distance)]
            if acceptance_x:
                x = [row["accepted_fraction_pct"] for row in selected]
            elif config["score_normalization"] == "minmax":
                x = [row["score_normalized"] for row in selected]
            elif config["score_normalization"] == "percentile":
                x = [row["score_percentile_pct"] for row in selected]
            else:
                x = [row["sc_dist_thres"] for row in selected]
            # Native thresholds can contain thousands of points; mark ~10 positions
            # instead of drawing a marker at every sample.
            x_values = np.asarray(x, dtype=float)
            markevery = sorted(set(int(np.argmin(np.abs(x_values - target))) for target in targets)) if x_values.size else []
            ax.plot(x, [row[field] for row in selected], label=run["label"],
                    markevery=markevery, **line_style(i))
        ax.set(xlabel=xlabel, ylabel=ylabel + " (%)", title=title, xlim=xlim, ylim=(-1, 101),
               yticks=np.arange(0, 101, 20))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        ax.tick_params(length=2.5, width=0.6, pad=2)
        ax.grid(True, color="0.87", linewidth=0.4)

    def save(fig, name):
        for extension in ("png", "pdf"):
            # Preserve physical dimensions, so 9 pt remains 9 pt at the intended paper width.
            fig.savefig(output / (name + "." + extension), dpi=600)
        plt.close(fig)

    def top_legend(fig, axes, wspace):
        axes = np.asarray(axes).ravel()
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.24, top=0.77, wspace=wspace)
        handles, labels = axes[0].get_legend_handles_labels()
        if not handles:
            return
        # Keep long algorithm names from forcing a single very wide row.
        labels = [textwrap.fill(label, width=28) for label in labels]
        legend = None
        for columns in range(min(len(labels), 4), 0, -1):
            if legend is not None:
                legend.remove()
            legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
                                ncol=columns, frameon=False, handlelength=1.6, handletextpad=0.5,
                                columnspacing=0.9, borderaxespad=0, labelspacing=0.35)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            box = legend.get_window_extent(renderer)
            if box.width <= fig.bbox.width * 0.98 or columns == 1:
                break
        title_height = max(max(0.0, ax.get_tightbbox(renderer).y1 - ax.bbox.y1) for ax in axes)
        gap = 10 * fig.dpi / 72  # Physical gap between legend and subplot titles.
        # Keep at least 45% of the figure for the axes; grow the figure if 6-8 rows need it.
        required = box.height + title_height + gap + 0.45 * fig.bbox.height + 0.24 * fig.bbox.height
        if required > fig.bbox.height:
            fig.set_size_inches(fig.get_figwidth(), required / fig.dpi)
            fig.canvas.draw()
            box = legend.get_window_extent(fig.canvas.get_renderer())
        top = (box.y0 - title_height - gap) / fig.bbox.height
        if top < 0.35:
            extra = (0.35 - top) * fig.bbox.height
            fig.set_size_inches(fig.get_figwidth(), (fig.bbox.height + extra) / fig.dpi)
            fig.canvas.draw()
            box = legend.get_window_extent(fig.canvas.get_renderer())
            top = (box.y0 - title_height - gap) / fig.bbox.height
        fig.subplots_adjust(top=max(0.30, min(0.92, top)))

    style = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
             "font.size": font, "axes.labelsize": font, "axes.titlesize": font,
             "xtick.labelsize": font - 1 if font > 1 else font, "ytick.labelsize": font - 1 if font > 1 else font,
             "legend.fontsize": legend_font, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        # One legend for the whole paper row, avoiding four repeated legends in narrow panels.
        fig, axes = plt.subplots(1, len(plots), figsize=(width, height), squeeze=False)
        for i, (ax, spec) in enumerate(zip(axes[0], plots)):
            panel(ax, spec, "(%s) %s" % (chr(97 + i), spec[4]))
        top_legend(fig, axes, wspace=0.48)
        save(fig, "comparison_row")

        # Individual panels keep the same font size at one-quarter of the paper-row width.
        panel_width = width / len(plots)
        wrap = max(12, int(panel_width * 72 / (legend_font * 0.55)) - 5)
        labels = [textwrap.fill(run["label"], width=wrap) for run in runs]
        legend_height = (sum(label.count("\n") + 1 for label in labels) + 1) * legend_font * 1.3 / 72
        for spec in plots:
            fig, ax = plt.subplots(figsize=(panel_width, height + legend_height))
            panel(ax, spec, spec[4])
            handles, _ = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                       frameon=False, handlelength=1.6, labelspacing=0.3)
            fig.subplots_adjust(left=0.27, right=0.93,
                                bottom=(legend_height + 0.4) / (height + legend_height),
                                top=1 - 0.25 / (height + legend_height))
            save(fig, spec[0])

        # Reparameterize the same score sweep by acceptance; do not average or resample.
        fig, axes = plt.subplots(1, len(distances), figsize=(width, height), sharey=True, squeeze=False)
        for i, (ax, distance) in enumerate(zip(axes[0], distances)):
            spec = ("", paired, "success_rate_pct", metric, "", distance)
            panel(ax, spec, "(%s) Recall@%g m" % (chr(97 + i), distance), acceptance_x=True)
            if i:
                ax.set_ylabel("")
        top_legend(fig, axes, wspace=0.25)
        save(fig, "recall_vs_acceptance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros", action="store_true", help="Force ROS mode (automatically detected under roslaunch)")
    for name, default in DEFAULTS.items():
        options = dict(default=default, type=type(default))
        if isinstance(default, list):
            options.update(type=float if name in ("recall_thresholds", "algorithm_score_maxima") else str, nargs="+")
        parser.add_argument("--" + name.replace("_", "-"), **options)
    args, unknown = parser.parse_known_args()
    # roslaunch supplies these remappings; no explicit flag is needed in the launch XML.
    if args.ros or any(arg.startswith(("__name:=", "__log:=")) for arg in unknown):
        import rospy
        rospy.init_node("localization_recall_comparison")
        config = {name: rospy.get_param("~" + name, default) for name, default in DEFAULTS.items()}
        try:
            compare(config)
        except (OSError, ValueError, TypeError) as error:
            rospy.logfatal("Recall comparison failed: %s", error)
            raise SystemExit(1)
    else:
        if unknown:
            parser.error("unrecognized arguments: " + " ".join(unknown))
        try:
            compare({name: getattr(args, name) for name in DEFAULTS})
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
