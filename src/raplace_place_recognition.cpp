// RaPlace-Light adapted to intensity/RCS BEV from 4D radar; shared ICP/CSV contracts.
#include "raplace.hpp"
#include "dyna_rc/raplace_input.h"
#include "dyna_rc/runtime_profiler.h"
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Geometry>
#include <fstream>
#include <memory>
#include <sstream>

class RaPlaceLocalization {
  using Cloud = raplace_input::Cloud;
  using Point = raplace_input::Point;
  struct Keyframe { int id; Eigen::Matrix4f pose; Cloud::Ptr cloud; };
  ros::NodeHandle nh_, pnh_{"~"};
  ros::Subscriber static_sub_, dynamic_sub_;
  ros::Publisher odom_pub_, index_pub_;
  tf::TransformBroadcaster tf_;
  sensor_msgs::PointCloud2ConstPtr static_msg_, dynamic_msg_;
  std::vector<Keyframe> map_;
  raplace::Params params_;
  std::unique_ptr<raplace::Descriptor> descriptor_;
  std::unique_ptr<raplace::Matcher> matcher_;
  localization_runtime::Profiler runtime_;
  std::string poses_file_, pcd_dir_, intensity_field_, map_intensity_field_, input_mode_;
  std::string map_frame_, radar_frame_;
  double radius_, z_min_, z_max_, threshold_, intensity_offset_, sync_tolerance_;
  double max_corr_dist_, trans_eps_, fitness_eps_;
  int oversample_, max_iter_, min_points_, queue_size_;
  bool publish_tf_;

  template<class T> T param(const std::string& name, const T& fallback) {
    T value;
    if (!pnh_.getParam(name, value)) nh_.param(name, value, fallback);
    pnh_.setParam(name, value);  // Snapshot resolved settings for runtime metadata.
    return value;
  }

  cv::Mat describe(const Cloud& cloud) {
    return descriptor_->fromCart(raplace_input::bev(cloud, params_.work_size, oversample_,
                                                  radius_, intensity_offset_));
  }

  Cloud::Ptr loadCloud(const std::string& path) {
    pcl::PCLPointCloud2 blob;
    if (pcl::io::loadPCDFile(path, blob) < 0) throw std::runtime_error("Cannot load " + path);
    return raplace_input::cloud(blob, map_intensity_field_, z_min_, z_max_);
  }

  void loadMap() {
    std::ifstream stream(poses_file_);
    if (!stream) throw std::runtime_error("Cannot open poses_file: " + poses_file_);
    ROS_INFO("Building RaPlace map descriptors from %s", poses_file_.c_str());
    int id = 0, missing_dynamic = 0, skipped = 0;
    for (std::string line; std::getline(stream, line); ) {
      if (line.find_first_not_of(" \t\r") == std::string::npos || line[line.find_first_not_of(" \t\r")] == '#') continue;
      std::istringstream row(line);
      double stamp, x, y, z, qx, qy, qz, qw;
      if (!(row >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw))
        throw std::runtime_error("Invalid map pose row " + std::to_string(id));
      Keyframe kf;
      kf.id = id++;  // Never renumber after skipping a missing/empty PCD.
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
      if (kf.cloud->size() < static_cast<size_t>(min_points_)) { ++skipped; continue; }
      const cv::Mat desc = describe(*kf.cloud);
      if (!cv::checkRange(desc) || cv::norm(desc, cv::NORM_L2SQR) <= 1e-12) { ++skipped; continue; }
      matcher_->add(desc);
      map_.push_back(std::move(kf));
      if (map_.size() % 500 == 0) ROS_INFO("RaPlace descriptors: %zu", map_.size());
    }
    if (map_.empty()) throw std::runtime_error("No usable map descriptors. Check PCD paths, intensity/RCS values, offset and range.");
    ROS_INFO("RaPlace map: %zu/%d poses searchable, skipped=%d, missing dynamic PCDs=%d; no recent-frame exclusion.",
             map_.size(), id, skipped, missing_dynamic);
    if (skipped || missing_dynamic)
      ROS_WARN("Check map completeness before comparing algorithms; original pose-file indices are preserved.");
  }

  Cloud::Ptr convert(const sensor_msgs::PointCloud2& msg) {
    pcl::PCLPointCloud2 blob;
    pcl_conversions::toPCL(msg, blob);
    return raplace_input::cloud(blob, intensity_field_, z_min_, z_max_);
  }

  void process() {
    if (!static_msg_ || (input_mode_ == "combined" && !dynamic_msg_)) return;
    if (input_mode_ == "combined") {
      const double delta = (static_msg_->header.stamp - dynamic_msg_->header.stamp).toSec();
      if (std::abs(delta) > sync_tolerance_) {
        ROS_WARN_THROTTLE(5, "RaPlace static/dynamic timestamp mismatch: %.3f s", delta);
        if (delta < 0) static_msg_.reset(); else dynamic_msg_.reset();
        return;
      }
    }
    const auto stat = static_msg_, dyna = dynamic_msg_;
    static_msg_.reset(); dynamic_msg_.reset();
    if (stat->header.stamp.isZero()) { ROS_WARN_THROTTLE(5, "RaPlace received zero timestamp"); return; }
    auto query = convert(*stat);
    if (dyna) {
      if (dyna->width && dyna->height && !stat->header.frame_id.empty() &&
          !dyna->header.frame_id.empty() && stat->header.frame_id != dyna->header.frame_id)
        throw std::runtime_error("Static and dynamic clouds must have the same sensor frame");
      *query += *convert(*dyna);
    }
    if (query->size() < static_cast<size_t>(min_points_)) {
      ROS_WARN_THROTTLE(5, "RaPlace: too few finite points after z filtering"); return;
    }
    localization_runtime::Profiler::Frame timing(runtime_, stat->header.stamp.toSec());
    localization_runtime::StageTimer stage(timing.sample.descriptor_ms);
    const cv::Mat desc = describe(*query);
    stage.next(timing.sample.retrieval_ms);
    int index = -1;
    double score = 0;
    const bool found = matcher_->query(desc, &index, &score);
    stage.stop();
    if (!found || !std::isfinite(score) || !(score < threshold_)) {
      ROS_WARN_THROTTLE(5, "RaPlace: no accepted match; check score gate, intensity/RCS and offset (score=%.4f)", score);
      return;
    }

    // Same point-to-point ICP and identity initialization as place_recognition.cpp.
    const auto started = localization_runtime::Clock::now();
    pcl::IterativeClosestPoint<Point, Point> icp;
    icp.setInputSource(query); icp.setInputTarget(map_.at(index).cloud);
    icp.setMaxCorrespondenceDistance(max_corr_dist_);
    icp.setMaximumIterations(max_iter_);
    icp.setTransformationEpsilon(trans_eps_);
    icp.setEuclideanFitnessEpsilon(fitness_eps_);
    Cloud aligned;
    icp.align(aligned);
    runtime_.addICP(localization_runtime::elapsedMs(started));
    if (!icp.hasConverged() || !icp.getFinalTransformation().allFinite()) {
      timing.sample.status = "icp_failed";
      ROS_WARN_THROTTLE(5, "RaPlace ICP failed; no localization output"); return;
    }
    const Eigen::Matrix4f pose = map_[index].pose * icp.getFinalTransformation();
    Eigen::Quaternionf q(pose.block<3,3>(0,0)); q.normalize();
    if (!pose.allFinite() || !q.coeffs().allFinite()) { timing.sample.status = "invalid_pose"; return; }
    nav_msgs::Odometry odom;
    odom.header.stamp = stat->header.stamp;  // Current observation, never map time or ros::Time::now().
    odom.header.frame_id = map_frame_; odom.child_frame_id = radar_frame_;
    odom.pose.pose.position.x = pose(0,3); odom.pose.pose.position.y = pose(1,3); odom.pose.pose.position.z = pose(2,3);
    odom.pose.pose.orientation.x = q.x(); odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z(); odom.pose.pose.orientation.w = q.w();
    odom.pose.covariance[0] = score;  // Native RaPlace distance, NOT cosine distance.
    odom.pose.covariance[1] = map_[index].id;
    odom_pub_.publish(odom);
    std_msgs::Int32 matched; matched.data = map_[index].id; index_pub_.publish(matched);
    if (publish_tf_) tf_.sendTransform(tf::StampedTransform(
        tf::Transform(tf::Quaternion(q.x(), q.y(), q.z(), q.w()), tf::Vector3(pose(0,3), pose(1,3), pose(2,3))),
        odom.header.stamp, map_frame_, radar_frame_));
    timing.sample.status = "published";
    timing.finish();
    ROS_INFO_THROTTLE(2, "RaPlace match index=%d, distance=%.4f", matched.data, score);
  }

public:
  RaPlaceLocalization() {
    poses_file_ = param<std::string>("poses_file", "poses.txt");
    pcd_dir_ = param<std::string>("pcd_dir", ".");
    input_mode_ = param<std::string>("raplace_input_mode", "combined");
    intensity_field_ = param<std::string>("raplace_intensity_field", "intensity");
    map_intensity_field_ = param<std::string>("raplace_map_intensity_field", intensity_field_);
    intensity_offset_ = param("raplace_intensity_offset", 0.0);
    radius_ = param("sc_max_radius", 250.0);
    z_min_ = param("z_min", -2.0); z_max_ = param("z_max", 50.0);
    threshold_ = param("sc_dist_thres", 1e6);
    sync_tolerance_ = param("raplace_sync_tolerance", 0.02);
    oversample_ = param("raplace_bev_oversample", 3);
    min_points_ = param("raplace_min_points", 3);
    queue_size_ = param("raplace_queue_size", 1);
    params_.work_size = param("raplace_work_size", 69);
    params_.n_theta = param("raplace_n_theta", 180);
    params_.n_theta_out = param("raplace_n_theta_out", 18);
    params_.exclude_recent = 0;  // Offline map, not incremental loop detection.
    max_corr_dist_ = param("max_corr_dist", 2.0); max_iter_ = param("max_iter", 50);
    trans_eps_ = param("trans_eps", 1e-6); fitness_eps_ = param("fitness_eps", 1e-6);
    map_frame_ = param<std::string>("map_frame", "map"); radar_frame_ = param<std::string>("radar_frame", "radar");
    publish_tf_ = param("publish_tf", true);
    const int threads = param("raplace_opencv_threads", 1);
    const std::string stat_topic = param<std::string>("radar_stat_topic", "/radar_points_inlier");
    const std::string dyna_topic = param<std::string>("radar_dyna_topic", "/radar_points_outlier");
    for (double v : {radius_, z_min_, z_max_, threshold_, intensity_offset_, sync_tolerance_, max_corr_dist_, trans_eps_, fitness_eps_})
      if (!std::isfinite(v)) throw std::runtime_error("RaPlace parameters must be finite");
    if ((input_mode_ != "combined" && input_mode_ != "static") || radius_ <= 0 || z_max_ <= z_min_ ||
        threshold_ <= 0 || sync_tolerance_ < 0 || oversample_ < 1 || oversample_ > 8 ||
        min_points_ < 3 || queue_size_ < 1 || threads < 1 || max_iter_ < 1 || max_corr_dist_ <= 0 ||
        trans_eps_ < 0 || fitness_eps_ < 0 || params_.work_size < 3 || params_.work_size > 1024 ||
        params_.n_theta < 1 || params_.n_theta_out < 1 || params_.n_theta_out > params_.n_theta)
      throw std::runtime_error("Invalid RaPlace/ICP parameters; input_mode must be combined or static");
    cv::setNumThreads(threads);
    descriptor_.reset(new raplace::Descriptor(params_)); matcher_.reset(new raplace::Matcher(params_));
    loadMap();
    runtime_.configure(pnh_, "RaPlace-4D-intensity", map_.size());
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/localization_odom", 10);
    index_pub_ = nh_.advertise<std_msgs::Int32>("/loop_closure_index", 10);
    static_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(stat_topic, queue_size_,
        [this](const sensor_msgs::PointCloud2ConstPtr& msg) { static_msg_ = msg; process(); });
    if (input_mode_ == "combined") dynamic_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(dyna_topic, queue_size_,
        [this](const sensor_msgs::PointCloud2ConstPtr& msg) { dynamic_msg_ = msg; process(); });
    ROS_INFO("RaPlace ready: %s points, field=%s, native distance gate=%.6g. Start bag playback now.",
             input_mode_.c_str(), intensity_field_.c_str(), threshold_);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "raplace_place_recognition");
  try { RaPlaceLocalization node; ros::spin(); }
  catch (const std::exception& error) { ROS_FATAL("RaPlace failed: %s", error.what()); return 1; }
  return 0;
}
