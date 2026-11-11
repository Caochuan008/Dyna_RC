#ifndef PLACE_RECOGNITION_H
#define PLACE_RECOGNITION_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>

#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "scancontext/Scancontext.h"   // SCManager 库
#include "dyna_rc/radar_point_type.h"
#include "dyna_rc/runtime_profiler.h"

using PointType = pcl::PointXYZI;

struct PointXYZIDynamic
{
    PCL_ADD_POINT4D;                  // 添加 x,y,z 字段
    float intensity;
    uint8_t dynamic;                  // 0: 静态, 1: 动态
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIDynamic,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint8_t, dynamic, dynamic)
)

class MapLocalization {
public:
  MapLocalization();
  void run();

private:

  std::ofstream log_file;
  ros::NodeHandle nh_;
  ros::Subscriber sub_radar_stat_;
  ros::Subscriber sub_radar_dyna_;
  ros::Publisher pub_odom_;
  ros::Publisher pub_loop_idx_;
  ros::Publisher pub_candidates_;
  ros::Publisher pub_candidate_scores_;
  ros::Publisher pub_combined_cloud_;

  // ===== GT pose (from /mavros/vision_pose/pose) =====
  ros::Subscriber sub_gt_pose_;

  Eigen::Vector3d current_gt_pos_;
  ros::Time current_gt_time_;
  bool has_gt_ = false;

  // ===== Recall =====
  int total_queries_ = 0;
  int correct_queries_ = 0;

  double recall_threshold_ = 150.0; // meters
  double sum_sq_error_ = 0.0;

  tf::TransformBroadcaster tf_broadcaster_;

  struct KeyFrame {
    int id;
    pcl::PointCloud<PointType>::Ptr cloud;       // 降采样后的地图点云（已在map坐标系下）
    Eigen::Vector3d position;                    // 地图坐标系下的位置
    Eigen::Quaterniond orientation;              // 地图坐标系下的姿态（四元数）
  };
  std::vector<KeyFrame> keyframes_;

  // 参数
  std::string poses_file_;
  std::string pcd_dir_;
  std::string radar_stat_topic_;
  std::string radar_dyna_topic_;
  std::string gt_topic_;
  int sc_channel_;
  double voxel_leaf_size_;
  double max_corr_dist_;
  int max_iter_;
  double trans_eps_;
  double fitness_eps_;
  double ndt_resolution_;
  double sc_dist_thres_;        // SC 匹配阈值
  double sc_max_radius_;        // SC 最大半径
  double z_min_;
  double z_max_;
  bool use_dynamic_sc_;

  SCManager sc_manager_;        // 扫描上下文管理器
  localization_runtime::Profiler runtime_;
  // 缓冲区，用于存储接收到的点云
  sensor_msgs::PointCloud2ConstPtr static_points_buffer_;
  sensor_msgs::PointCloud2ConstPtr dynamic_points_buffer_;

  void loadMap();
  bool alignClouds(pcl::PointCloud<PointType>::Ptr source,
                   pcl::PointCloud<PointType>::Ptr target,
                   Eigen::Matrix4f& transform);
  void staticPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void gtPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void dynamicPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void processPoints();
  void publishCandidatePoses(const std::vector<std::pair<int, float>>& candidates,
    pcl::PointCloud<PointType>::Ptr current_cloud,
    const ros::Time& stamp);
};

#endif // PLACE_RECOGNITION_H
