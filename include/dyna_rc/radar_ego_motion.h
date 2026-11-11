#ifndef PLACE_RECOGNITION_RADAR_EGO_MOTION_H
#define PLACE_RECOGNITION_RADAR_EGO_MOTION_H
#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include "dyna_rc/radar_point_type.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/plane_clipper3D.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

class RadarEgoMotion {
 public:
  RadarEgoMotion();
  void ScanCallback(const sensor_msgs::PointCloud2ConstPtr& scan);
  bool msg_to_pcl(const sensor_msgs::PointCloud2& msg, pcl::PointCloud<RadarPointType>::Ptr cloud);
  void doppler_ego_motion(const pcl::PointCloud<RadarPointType>& cloud,
                         Eigen::VectorXf* doppler, Eigen::MatrixXf* directions);
  std::vector<int> Ransac(const Eigen::VectorXf& doppler, const Eigen::MatrixXf& directions,
                          double threshold);
  std::vector<int> Ransac(const Eigen::VectorXf& doppler, const Eigen::MatrixXf& directions,
                          int sample_count, int iterations, double threshold);
  void ground_filter(const pcl::PointCloud<RadarPointType>& cloud,
                     pcl::PointCloud<RadarPointType>::Ptr filtered, double sensor_height);
  void unstable_point_detector(const geometry_msgs::TwistWithCovarianceStamped& velocity,
                               pcl::PointCloud<RadarPointType>& inliers);

 private:
  template<class T> void param(const std::string& key, T& value, const T& fallback) {
    if (!pnh_.getParam(key, value)) nh_.param(key, value, fallback);
  }
  int calc_ransac_iter(int samples) const {
    return static_cast<int>(std::log(1.0 - 0.9999) / std::log(1.0 - std::pow(0.6, samples)));
  }
  Eigen::MatrixXd calc_cov(const Eigen::MatrixXd& a, double accuracy) const {
    // Preserve the original diagonal information approximation, but avoid infinities.
    const Eigen::MatrixXd info = a.transpose() * a / accuracy;
    return info.diagonal().cwiseMax(1e-12).cwiseInverse().asDiagonal();
  }
  void publishEmptyClouds();

  ros::NodeHandle nh_, pnh_{"~"};
  ros::Subscriber sub_scan_;
  ros::Publisher pub_ego_vel_, pub_outlier_, pub_inlier_, pub_radar_vel_, pub_inlier_2d_;
  std_msgs::Header msg_header;
  int n_dof_, ransac_iter_, radar_type_, total_frames_ = 0;
  double t_x, t_y, t_z, mounting_angle_x, mounting_angle_y, mounting_angle_z;
  double min_height, max_height, min_intensity, max_scan_range, min_scan_range;
  double velocity_accuracy, ransac_threshold, input_leaf_size_, total_ego_vel_time_ = 0;
  double unstable_radius_, unstable_max_dt_;
  bool debug, tictoc, unstable_filter, use_rcs, ground_filter_enabled_, initial = false;
  std::string rad_topic;
  geometry_msgs::TwistWithCovarianceStamped last_radar_vel, last_ego_vel;
  pcl::PointCloud<RadarPointType> last_inliers;
  std::ofstream log_file;
};
#endif
