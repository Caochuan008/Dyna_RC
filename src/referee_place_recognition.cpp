// ReFeree polar feature/free-space descriptors + existing point-to-point ICP.
#include "dyna_rc/referee_input.h"
#include "dyna_rc/runtime_profiler.h"
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

class ReFereeLocalization {
  using Cloud = referee_input::Cloud;
  using Point = referee_input::Point;
  struct Keyframe { int id; Eigen::Matrix4f pose; Cloud::Ptr cloud; referee::Descriptor descriptor; };
  ros::NodeHandle nh_, pnh_{"~"};
  ros::Subscriber static_sub_, dynamic_sub_;
  ros::Publisher odom_pub_, index_pub_;
  tf::TransformBroadcaster tf_;
  sensor_msgs::PointCloud2ConstPtr static_msg_, dynamic_msg_;
  std::vector<Keyframe> map_;
  referee::Params params_;
  localization_runtime::Profiler runtime_;
  std::string poses_file_, pcd_dir_, intensity_field_, map_intensity_field_, input_mode_;
  std::string map_frame_, radar_frame_;
  double radius_, z_min_, z_max_, threshold_, min_intensity_, sync_tolerance_;
  double max_corr_dist_, trans_eps_, fitness_eps_;
  int max_iter_, min_points_, queue_size_;
  bool publish_tf_, use_yaw_;

  template<class T> T param(const std::string& name, const T& fallback) {
    T value;
    if (!pnh_.getParam(name, value)) nh_.param(name, value, fallback);
    pnh_.setParam(name, value);
    return value;
  }

  referee::Descriptor describe(const Cloud& cloud) const {
    return referee_input::describe(cloud, params_, radius_, min_intensity_);
  }

  Cloud::Ptr readCloud(const pcl::PCLPointCloud2& blob, const std::string& field) const {
    return raplace_input::cloud(blob, field, z_min_, z_max_);
  }

  Cloud::Ptr loadCloud(const std::string& path) const {
    pcl::PCLPointCloud2 blob;
    if (pcl::io::loadPCDFile(path, blob) < 0) throw std::runtime_error("Cannot load " + path);
    return readCloud(blob, map_intensity_field_);
  }

  void loadMap() {
    std::ifstream stream(poses_file_);
    if (!stream) throw std::runtime_error("Cannot open poses_file: " + poses_file_);
    ROS_INFO("Building ReFeree map descriptors from %s", poses_file_.c_str());
    int id = 0, missing_dynamic = 0, skipped = 0;
    for (std::string line; std::getline(stream, line); ) {
      const auto first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos || line[first] == '#') continue;
      std::istringstream row(line);
      double stamp, x, y, z, qx, qy, qz, qw;
      if (!(row >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw))
        throw std::runtime_error("Invalid map pose row " + std::to_string(id));
      Keyframe kf;
      kf.id = id++;  // Original pose-file index, even when earlier PCDs are missing.
      Eigen::Quaterniond q(qw, qx, qy, qz);
      Eigen::Vector3d t(x, y, z);
      if (!t.allFinite() || !q.coeffs().allFinite() || q.norm() < 1e-12 || !std::isfinite(stamp))
        throw std::runtime_error("Invalid map pose at index " + std::to_string(kf.id));
      kf.pose = Eigen::Matrix4f::Identity();
      kf.pose.block<3,3>(0,0) = q.normalized().toRotationMatrix().cast<float>();
      kf.pose.block<3,1>(0,3) = t.cast<float>();
      const std::string prefix = pcd_dir_ + "/" + std::to_string(kf.id);
      if (!std::filesystem::exists(prefix + ".pcd")) { ++skipped; continue; }
      kf.cloud = loadCloud(prefix + ".pcd");
      if (input_mode_ == "combined") {
        if (std::filesystem::exists(prefix + "_dyna.pcd")) *kf.cloud += *loadCloud(prefix + "_dyna.pcd");
        else ++missing_dynamic;
      }
      kf.descriptor = describe(*kf.cloud);
      if (!kf.descriptor.valid() || kf.descriptor.occupied < static_cast<size_t>(min_points_)) { ++skipped; continue; }
      map_.push_back(std::move(kf));
      if (map_.size() % 500 == 0) ROS_INFO("ReFeree descriptors: %zu", map_.size());
    }
    if (map_.empty()) throw std::runtime_error("No usable ReFeree map descriptors; check PCD paths, range and filters");
    ROS_INFO("ReFeree map: %zu/%d searchable; skipped=%d, missing dynamic PCDs=%d; R=%zu, A=%zu",
             map_.size(), id, skipped, missing_dynamic, map_[0].descriptor.range.size(), map_[0].descriptor.angle.size());
    if (skipped || missing_dynamic) ROS_WARN("Check map completeness for comparisons; original indices are preserved");
  }

  Cloud::Ptr convert(const sensor_msgs::PointCloud2& msg) const {
    pcl::PCLPointCloud2 blob;
    pcl_conversions::toPCL(msg, blob);
    return readCloud(blob, intensity_field_);
  }

  void process() {
    if (!static_msg_ || (input_mode_ == "combined" && !dynamic_msg_)) return;
    if (input_mode_ == "combined") {
      const double delta = (static_msg_->header.stamp - dynamic_msg_->header.stamp).toSec();
      if (std::abs(delta) > sync_tolerance_) {
        ROS_WARN_THROTTLE(5, "ReFeree static/dynamic timestamp mismatch: %.3f s", delta);
        if (delta < 0) static_msg_.reset(); else dynamic_msg_.reset();
        return;
      }
    }
    const auto stat = static_msg_, dyna = dynamic_msg_;
    static_msg_.reset(); dynamic_msg_.reset();
    if (stat->header.stamp.isZero()) { ROS_WARN_THROTTLE(5, "ReFeree received zero timestamp"); return; }
    auto query = convert(*stat);
    if (dyna) {
      if (dyna->width && dyna->height && !stat->header.frame_id.empty() &&
          !dyna->header.frame_id.empty() && stat->header.frame_id != dyna->header.frame_id)
        throw std::runtime_error("Static and dynamic clouds must have the same sensor frame");
      *query += *convert(*dyna);
    }
    if (query->size() < static_cast<size_t>(min_points_)) {
      ROS_WARN_THROTTLE(5, "ReFeree: too few finite points after z filtering"); return;
    }
    localization_runtime::Profiler::Frame timing(runtime_, stat->header.stamp.toSec());
    localization_runtime::StageTimer stage(timing.sample.descriptor_ms);
    const auto desc = describe(*query);
    if (!desc.valid() || desc.occupied < static_cast<size_t>(min_points_)) {
      timing.sample.status = "empty_descriptor";
      ROS_WARN_THROTTLE(5, "ReFeree: too few occupied polar cells; check range and intensity filters"); return;
    }
    stage.next(timing.sample.retrieval_ms);
    size_t index = 0;
    double score = std::numeric_limits<double>::infinity();
    // Exact search over the fixed map. No exclusion of its final keyframes.
    for (size_t i = 0; i < map_.size(); ++i) {
      const double d = referee::distance(desc, map_[i].descriptor);
      if (d < score) { score = d; index = i; }
    }
    if (!std::isfinite(score) || !(score < threshold_)) {
      ROS_WARN_THROTTLE(5, "ReFeree L2 distance %.6g rejected by threshold %.6g", score, threshold_); return;
    }
    const double yaw = use_yaw_ ? referee::queryToMapYaw(desc, map_[index].descriptor) : 0.0;
    stage.stop();

    const auto started = localization_runtime::Clock::now();
    Eigen::Matrix4f initial = Eigen::Matrix4f::Identity();
    initial.block<3,3>(0,0) = Eigen::AngleAxisf(static_cast<float>(yaw), Eigen::Vector3f::UnitZ()).toRotationMatrix();
    pcl::IterativeClosestPoint<Point, Point> icp;
    icp.setInputSource(query); icp.setInputTarget(map_[index].cloud);
    icp.setMaxCorrespondenceDistance(max_corr_dist_); icp.setMaximumIterations(max_iter_);
    icp.setTransformationEpsilon(trans_eps_); icp.setEuclideanFitnessEpsilon(fitness_eps_);
    Cloud aligned;
    icp.align(aligned, initial);  // getFinalTransformation already INCLUDES initial.
    runtime_.addICP(localization_runtime::elapsedMs(started));
    if (!icp.hasConverged() || !icp.getFinalTransformation().allFinite()) {
      timing.sample.status = "icp_failed";
      ROS_WARN_THROTTLE(5, "ReFeree ICP failed; no localization output"); return;
    }
    const Eigen::Matrix4f pose = map_[index].pose * icp.getFinalTransformation();
    Eigen::Quaternionf q(pose.block<3,3>(0,0)); q.normalize();
    if (!pose.allFinite() || !q.coeffs().allFinite()) { timing.sample.status = "invalid_pose"; return; }
    nav_msgs::Odometry odom;
    odom.header.stamp = stat->header.stamp;
    odom.header.frame_id = map_frame_; odom.child_frame_id = radar_frame_;
    odom.pose.pose.position.x = pose(0,3); odom.pose.pose.position.y = pose(1,3); odom.pose.pose.position.z = pose(2,3);
    odom.pose.pose.orientation.x = q.x(); odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z(); odom.pose.pose.orientation.w = q.w();
    odom.pose.covariance[0] = score;  // Raw R-ReFeree L2 distance, not SC cosine distance.
    odom.pose.covariance[1] = map_[index].id;
    odom_pub_.publish(odom);
    std_msgs::Int32 matched; matched.data = map_[index].id; index_pub_.publish(matched);
    if (publish_tf_) tf_.sendTransform(tf::StampedTransform(
        tf::Transform(tf::Quaternion(q.x(), q.y(), q.z(), q.w()), tf::Vector3(pose(0,3), pose(1,3), pose(2,3))),
        odom.header.stamp, map_frame_, radar_frame_));
    timing.sample.status = "published";
    timing.finish();
    ROS_INFO_THROTTLE(2, "ReFeree index=%d, L2=%.6g, query-to-map yaw=%.1f deg", matched.data, score, yaw*180/referee::pi);
  }

public:
  ReFereeLocalization() {
    poses_file_ = param<std::string>("poses_file", "poses.txt"); pcd_dir_ = param<std::string>("pcd_dir", ".");
    input_mode_ = param<std::string>("referee_input_mode", "combined");
    intensity_field_ = param<std::string>("referee_intensity_field", "intensity");
    map_intensity_field_ = param<std::string>("referee_map_intensity_field", intensity_field_);
    min_intensity_ = param("referee_min_intensity", -1e30);
    radius_ = param("sc_max_radius", 250.0);
    z_min_ = param("z_min", -2.0); z_max_ = param("z_max", 50.0);
    threshold_ = param("sc_dist_thres", 1e9);
    sync_tolerance_ = param("referee_sync_tolerance", 0.02);
    min_points_ = param("referee_min_points", 3); queue_size_ = param("referee_queue_size", 1);
    params_.angle_bins = param("referee_angle_bins", 400); params_.range_bins = param("referee_range_bins", 3360);
    params_.range_block = param("referee_range_block", 80); params_.angle_block = param("referee_angle_block", 8);
    use_yaw_ = param("referee_use_yaw", true);
    max_corr_dist_ = param("max_corr_dist", 2.0); max_iter_ = param("max_iter", 50);
    trans_eps_ = param("trans_eps", 1e-6); fitness_eps_ = param("fitness_eps", 1e-6);
    map_frame_ = param<std::string>("map_frame", "map"); radar_frame_ = param<std::string>("radar_frame", "radar");
    publish_tf_ = param("publish_tf", true);
    const std::string stat_topic = param<std::string>("radar_stat_topic", "/radar_points_inlier");
    const std::string dyna_topic = param<std::string>("radar_dyna_topic", "/radar_points_outlier");
    for (double v : {radius_, z_min_, z_max_, threshold_, min_intensity_, sync_tolerance_, max_corr_dist_, trans_eps_, fitness_eps_})
      if (!std::isfinite(v)) throw std::runtime_error("ReFeree parameters must be finite");
    params_.validate();
    if ((input_mode_ != "combined" && input_mode_ != "static") || radius_ <= 0 || z_max_ <= z_min_ ||
        threshold_ <= 0 || sync_tolerance_ < 0 || min_points_ < 3 || queue_size_ < 1 || max_iter_ < 1 ||
        max_corr_dist_ <= 0 || trans_eps_ < 0 || fitness_eps_ < 0 || stat_topic.empty() ||
        (input_mode_ == "combined" && (dyna_topic.empty() || nh_.resolveName(stat_topic) == nh_.resolveName(dyna_topic))))
      throw std::runtime_error("Invalid ReFeree/ICP parameters or static/dynamic topics");
    loadMap();
    runtime_.configure(pnh_, "ReFeree-4D-L2-ICP", map_.size());
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/localization_odom", 10);
    index_pub_ = nh_.advertise<std_msgs::Int32>("/loop_closure_index", 10);
    static_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(stat_topic, queue_size_,
        [this](const sensor_msgs::PointCloud2ConstPtr& msg) { static_msg_ = msg; process(); });
    if (input_mode_ == "combined") dynamic_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(dyna_topic, queue_size_,
        [this](const sensor_msgs::PointCloud2ConstPtr& msg) { dynamic_msg_ = msg; process(); });
    ROS_INFO("ReFeree ready: %s, field=%s, L2 gate=%.6g, yaw init=%d. Start bag playback now.",
             input_mode_.c_str(), intensity_field_.c_str(), threshold_, use_yaw_);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "referee_place_recognition");
  try { ReFereeLocalization node; ros::spin(); }
  catch (const std::exception& error) { ROS_FATAL("ReFeree failed: %s", error.what()); return 1; }
  return 0;
}
