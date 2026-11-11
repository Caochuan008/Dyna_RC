#!/usr/bin/env python3
"""Record scored localization results and timestamp-aligned errors for offline evaluation."""
import csv
import hashlib
import json
import threading
from collections import Counter
from pathlib import Path

import numpy as np


def load_poses(path):
    poses = np.loadtxt(path, comments="#", usecols=range(8), ndmin=2)
    if not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Empty or non-finite trajectory: " + str(path))
    norms = np.linalg.norm(poses[:, 4:8], axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Zero quaternion in " + str(path))
    poses[:, 4:8] /= norms[:, None]
    return poses  # Map row order is the covariance[1] index; never sort the map.


def yaw(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        return np.nan
    x, y, z, w = q / np.linalg.norm(q)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class GroundTruth:
    def __init__(self, poses, max_gap=1.0):
        if not np.isfinite(max_gap) or max_gap < 0:
            raise ValueError("max_gt_gap must be finite and nonnegative")
        poses = poses[np.argsort(poses[:, 0], kind="stable")]
        for i in np.flatnonzero(np.diff(poses[:, 0]) == 0):
            if (not np.allclose(poses[i, 1:4], poses[i + 1, 1:4], rtol=0, atol=1e-6)
                    or abs(np.dot(poses[i, 4:8], poses[i + 1, 4:8])) < 1 - 1e-10):
                raise ValueError("Conflicting GT poses at timestamp %.9f" % poses[i, 0])
        self.poses = poses[np.r_[True, np.diff(poses[:, 0]) > 0]]
        if len(self.poses) < 2:
            raise ValueError("GT requires at least two distinct timestamps")
        self.times, self.max_gap = self.poses[:, 0], max_gap

    def interpolate(self, stamp):
        j = int(np.searchsorted(self.times, stamp))
        if j < len(self.times) and self.times[j] == stamp:
            return self.poses[j, 1:4], self.poses[j, 4:8]
        if j == 0 or j == len(self.times):
            raise ValueError("outside_gt_time_range")
        a, b = self.poses[j - 1], self.poses[j]
        gap = b[0] - a[0]
        if self.max_gap > 0 and gap > self.max_gap:
            raise ValueError("gt_interpolation_gap")
        alpha = (stamp - a[0]) / gap
        q0, q1 = a[4:8], b[4:8].copy()
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1, dot = -q1, -dot
        if dot > 0.9995:
            q = (1 - alpha) * q0 + alpha * q1
        else:
            angle = np.arccos(np.clip(dot, -1, 1))
            q = (np.sin((1 - alpha) * angle) * q0 + np.sin(alpha * angle) * q1) / np.sin(angle)
        return a[1:4] + alpha * (b[1:4] - a[1:4]), q / np.linalg.norm(q)


FIELDS = ("stamp_ns", "stamp", "gt_stamp", "score", "matched_index", "matched_index_valid", "gt_valid", "status",
          "gt_x", "gt_y", "gt_z", "gt_yaw_deg", "matched_x", "matched_y", "matched_z", "matched_yaw_deg",
          "localization_x", "localization_y", "localization_z", "localization_yaw_deg",
          "matched_xy_error_m", "matched_xyz_error_m", "matched_yaw_error_deg",
          "localization_xy_error_m", "localization_xyz_error_m", "localization_yaw_error_deg")


def make_record(msg, keyframes, gt, time_offset=0.0):
    stamp_ns = int(msg.header.stamp.to_nsec())
    stamp = msg.header.stamp.to_sec()
    if stamp_ns <= 0:
        raise ValueError("zero_or_negative_timestamp")
    score, raw_index = map(float, msg.pose.covariance[:2])
    valid_index = (np.isfinite(raw_index) and abs(raw_index - round(raw_index)) < 1e-3
                   and 0 <= round(raw_index) < len(keyframes))
    row = dict.fromkeys(FIELDS, np.nan)
    row.update(stamp_ns=stamp_ns, stamp="%.9f" % stamp, gt_stamp="%.9f" % (stamp + time_offset),
               score=score, matched_index=raw_index, matched_index_valid=int(valid_index), gt_valid=0, status="ok")
    pose = msg.pose.pose
    estimate = np.array([pose.position.x, pose.position.y, pose.position.z])
    q = pose.orientation
    estimate_yaw = yaw([q.x, q.y, q.z, q.w])
    row.update(zip(("localization_x", "localization_y", "localization_z"), estimate))
    row["localization_yaw_deg"] = np.degrees(estimate_yaw)
    if valid_index:
        frame = keyframes[int(round(raw_index))]
        row.update(zip(("matched_x", "matched_y", "matched_z"), frame[1:4]))
        row["matched_yaw_deg"] = np.degrees(yaw(frame[4:8]))
    try:
        position, quaternion = gt.interpolate(stamp + time_offset)
    except ValueError as error:
        row["status"] = str(error)
        return row  # Preserve the observation; plotting excludes rows without evaluable GT.
    gt_yaw = yaw(quaternion)
    row.update(gt_valid=1, gt_yaw_deg=np.degrees(gt_yaw))
    row.update(zip(("gt_x", "gt_y", "gt_z"), position))
    for prefix in ("matched", "localization"):
        delta = np.array([row[prefix + "_" + axis] for axis in "xyz"]) - position
        angle = np.radians(row[prefix + "_yaw_deg"]) - gt_yaw
        row[prefix + "_xy_error_m"] = np.linalg.norm(delta[:2])
        row[prefix + "_xyz_error_m"] = np.linalg.norm(delta)
        row[prefix + "_yaw_error_deg"] = np.degrees(abs(np.arctan2(np.sin(angle), np.cos(angle))))
    issues = []
    if not np.isfinite(score) or score < 0:
        issues.append("invalid_score")
    if not valid_index:
        issues.append("invalid_matched_index")
    if not np.isfinite(estimate).all() or not np.isfinite(estimate_yaw):
        issues.append("invalid_output_pose")
    row["status"] = ";".join(issues) or "ok"
    return row


class RecallRecorder:
    def __init__(self):
        import rospy
        from nav_msgs.msg import Odometry
        self.ros, self.lock = rospy, threading.RLock()
        param = rospy.get_param
        map_path = param("~keyframe_poses_file", "") or param("/poses_file", "")
        if not map_path:
            raise ValueError("Set ~keyframe_poses_file to the first-lap map pose file")
        map_path = Path(map_path).expanduser().resolve()
        gt_file = param("~gt_file", "")
        if not gt_file:
            raise ValueError("Set ~gt_file to a trajectory in the map coordinate frame")
        gt_path = Path(gt_file).expanduser().resolve()
        self.keyframes = load_poses(map_path)
        self.gt = GroundTruth(load_poses(gt_path), float(param("~max_gt_gap", 1.0)))
        self.time_offset = float(param("~gt_time_offset", 0.0))
        if not np.isfinite(self.time_offset):
            raise ValueError("gt_time_offset must be finite")
        self.output = Path(param("~output_csv", "~/.ros/localization_recall/alg1.csv")).expanduser().resolve()
        if self.output.suffix.lower() != ".csv":
            raise ValueError("output_csv must have a .csv extension")
        self.meta_path = self.output.with_suffix(".meta.json")
        overwrite = param("~overwrite", False)
        if not overwrite and (self.output.exists() or self.meta_path.exists()):
            raise ValueError("Output already exists; choose another output_csv or set ~overwrite:=true")
        topic = param("~localization_topic", "/localization_odom")
        threshold_param = param("~score_threshold_param", "") or (
            "/seq_distance_threshold" if topic.strip("/").endswith("sequence/localization_odom") else "/sc_dist_thres")
        self.metadata = dict(schema="localization_recall_v1", algorithm_name=param("~algorithm_name", "") or self.output.stem,
            keyframe_poses_file=str(map_path), gt_file=str(gt_path), localization_topic=topic,
            map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest(),
            gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(), gt_time_offset=self.time_offset,
            max_gt_gap=self.gt.max_gap, producer_sc_dist_thres=param(threshold_param, None),
            producer_score_threshold_param=threshold_param,
            score_definition=param("~score_definition", "producer-specific distance; lower is better"),
            timestamp_reference="localization header.stamp; original integer nanoseconds retained",
            query_scope="published localization results only; missing producer outputs are not observed",
            status="recording")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output.open("w" if overwrite else "x", newline="", buffering=1, encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.received, self.closed, self.seen, self.counts = 0, False, set(), Counter()
        self.write_metadata()
        rospy.on_shutdown(self.close)
        self.sub = rospy.Subscriber(topic, Odometry, self.callback, queue_size=1000)
        rospy.loginfo("Recall recorder: %d map poses; output=%s; GT=[%.9f, %.9f]",
                      len(self.keyframes), self.output, self.gt.times[0], self.gt.times[-1])
        rospy.logwarn("Recording gate %s=%s must cover the intended score sweep. Alignment failures still produce no message.",
                      threshold_param, self.metadata["producer_sc_dist_thres"])

    def write_metadata(self):
        self.metadata.update(received=self.received, recorded=len(self.seen), counters=dict(self.counts))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    def callback(self, msg):
        with self.lock:
            if self.closed:
                return
            self.received += 1
            stamp_ns = int(msg.header.stamp.to_nsec())
            if stamp_ns in self.seen:
                self.counts["duplicate_stamp"] += 1
                return
            try:
                row = make_record(msg, self.keyframes, self.gt, self.time_offset)
            except ValueError as error:
                self.counts[str(error)] += 1
                return
            self.writer.writerow(row)
            self.seen.add(stamp_ns)
            self.counts[row["status"]] += 1
            if not row["gt_valid"]:
                self.ros.logwarn_throttle(5, "Recall record has no aligned GT: " + row["status"])
            self.ros.loginfo_throttle(10, "Recall recorder: %d unique observations saved to %s" % (len(self.seen), self.output))

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.file.close()
            self.metadata["status"] = "complete"
            self.write_metadata()
            self.ros.loginfo("Saved %d observations to %s", len(self.seen), self.output)


if __name__ == "__main__":
    import rospy
    rospy.init_node("localization_recall_recorder")
    try:
        recorder = RecallRecorder()
    except (OSError, ValueError) as error:
        rospy.logfatal("Cannot start recall recorder: %s", error)
        raise SystemExit(1)
    rospy.spin()
