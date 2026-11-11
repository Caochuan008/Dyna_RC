#include "dyna_rc/seq_place_recognition.h"
#include <cmath>

SeqMapLocalization::SeqMapLocalization() {
  // 读取参数
  nh_.param("poses_file", poses_file_, std::string("poses.txt"));
  nh_.param("pcd_dir", pcd_dir_, std::string("."));
  nh_.param("voxel_leaf_size", voxel_leaf_size_, 0.2);
  nh_.param("max_corr_dist", max_corr_dist_, 2.0);
  nh_.param("max_iter", max_iter_, 50);
  nh_.param("trans_eps", trans_eps_, 1e-6);
  nh_.param("fitness_eps", fitness_eps_, 1e-6);
  nh_.param("ndt_resolution", ndt_resolution_, 1.0);
  nh_.param("sc_dist_thres", sc_dist_thres_, 0.2);
  nh_.param("sc_max_radius", sc_max_radius_, 80.0);
  nh_.param("radar_topic", radar_topic_, std::string("/radar_v_points"));
  nh_.param("radar_stat_topic", radar_stat_topic_, std::string("/radar_points_inlier"));
  nh_.param("radar_dyna_topic", radar_dyna_topic_, std::string("/radar_points_outlier"));
  nh_.param("gt_topic", gt_topic_, std::string("/mavros/vision_pose/pose"));
  nh_.param("use_dynamic_sc", use_dynamic_sc_, false);  // 新增：是否使用动态SC
  nh_.param("seq_type", seq_type, 1);
  nh_.param("seq_distance_threshold", seq_distance_threshold_, 0.4);
  nh_.param("recall_threshold", recall_threshold_, 150.0);
  nh_.param("z_min", z_min_, -2.0);
  nh_.param("z_max", z_max_, 50.0); 
  nh_.param("sc_channel", sc_channel_, 1);  //1:z  2:intensity
  nh_.param("skip_fine_matching", skip_fine_matching_, false);


  // 配置 SCManager
  seq_relocalizer_.setSCdistThres(sc_dist_thres_);
  seq_relocalizer_.setScChannel(sc_channel_);
  seq_relocalizer_.setMaximumRadius(sc_max_radius_);
  seq_relocalizer_.setUseDynamicSC(use_dynamic_sc_);
  seq_relocalizer_.setParams(
        8,   // fine window size
        8,   // exclude recent frames
        16,    // coarse window size
        30,   // fine window size (again, 可根据需要)
        seq_distance_threshold_  // distance threshold
    );
  seq_relocalizer_.setVotingParams(
        5,   // top_k_single_frame
        5,   // top_m_coarse_candidates
        3,   // candidate_tolerance
        skip_fine_matching_
  );

  // 加载地图
  loadMap();
  const std::string method = std::string(use_dynamic_sc_ ? "Seq-RC" : "Seq-SC-z") +
      (seq_type == 1 ? "-FPM" : "-ERA") + (skip_fine_matching_ ? "-no-fine" : "");
  runtime_.configure(nh_, method, keyframes_.size(), seq_relocalizer_.getFineWindowSize(), 5);
  ROS_INFO("Sequence database: %zu descriptors, %zu masks.",
           seq_relocalizer_.polarcontexts_.size(), seq_relocalizer_.dynamic_masks_.size());

  // 按模式订阅：普通SC使用原始点云，动态SC使用前端分离后的两路点云。
  if (use_dynamic_sc_) {
    sub_radar_stat_ = nh_.subscribe(radar_stat_topic_, 1, &SeqMapLocalization::staticPointsCallback, this);
    sub_radar_dyna_ = nh_.subscribe(radar_dyna_topic_, 1, &SeqMapLocalization::dynamicPointsCallback, this);
    ROS_INFO("Sequence dynamic SC: static=%s, dynamic=%s", radar_stat_topic_.c_str(), radar_dyna_topic_.c_str());
  } else {
    sub_radar_ = nh_.subscribe(radar_topic_, 1, &SeqMapLocalization::radarCallback, this);
    ROS_INFO("Sequence original SC: raw=%s", radar_topic_.c_str());
  }
  sub_gt_pose_ = nh_.subscribe(gt_topic_, 10, &SeqMapLocalization::gtPoseCallback, this);
  // 发布定位结果和回环序号
  pub_odom_ = nh_.advertise<nav_msgs::Odometry>("/sequence/localization_odom", 10);
  pub_odom__camera_init = nh_.advertise<nav_msgs::Odometry>("/sequence/localization_odom_camera_init", 10);
  pub_loop_idx_ = nh_.advertise<std_msgs::Int32>("/sequence/loop_closure_index", 10);

  if (use_dynamic_sc_) {
    ROS_INFO("SeqMapLocalization initialized with dynamic SC, loaded %lu keyframes.", keyframes_.size());
  } else {
    ROS_INFO("SeqMapLocalization initialized with original SC, loaded %lu keyframes.", keyframes_.size());
  }
  std::string log_path;
  nh_.param("metrics_log_file", log_path, std::string());
  if (!log_path.empty()) {
    log_file.open(log_path, std::ios::out | std::ios::app);
    if (!log_file.is_open()) ROS_WARN("Cannot open metrics_log_file: %s", log_path.c_str());
  }
}

void SeqMapLocalization::loadMap() {
  std::ifstream file(poses_file_);
  if (!file.is_open()) {
    ROS_ERROR("Failed to open poses file: %s", poses_file_.c_str());
    return;
  }

  std::string line;
  int id=0;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    
    double timestamp, x, y, z, qx, qy, qz, qw;
    if (!(ss >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw)) {
      ROS_WARN("Invalid line in poses file: %s", line.c_str());
      continue;
    }
    // std::cout<<"timestamp, x,y,z,qx,qy,qz,qw:"<<timestamp<<","<<x<<","<<y<<","<<z<<","<<qx<<","<<qy<<","<<qz<<","<<qw<<std::endl;
    KeyFrame kf;
    kf.id = id++;
    kf.position = Eigen::Vector3d(x, y, z);
    kf.orientation = Eigen::Quaterniond(qw, qx, qy, qz);  // 注意四元数顺序 (w, x, y, z)

    // 加载对应的PCD文件
    std::string pcd_path = pcd_dir_ + "/" + std::to_string(kf.id) + ".pcd";

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>(pcd_path, *cloud) == -1) {
      ROS_ERROR("Failed to load PCD file: %s", pcd_path.c_str());
      continue;
    }

    // 按 z 值过滤
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    cloud_filtered->reserve(cloud->size());
    for (const auto& pt : cloud->points) {
      if (pt.z >= z_min_ && pt.z <= z_max_) {
        cloud_filtered->push_back(pt);
      }
    }
    cloud_filtered->width  = cloud_filtered->size();
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = cloud->is_dense;

    // 两种模式都读取动态PCD；它与静态PCD共同组成原始帧。
    std::string pcd_dyna_path = pcd_dir_ + "/" + std::to_string(kf.id) + "_dyna.pcd";
    pcl::PointCloud<PointType>::Ptr dyna_cloud(new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType>(pcd_dyna_path, *dyna_cloud) == -1) {
      ROS_WARN("Failed to load dynamic PCD file: %s, using an empty dynamic cloud", pcd_dyna_path.c_str());
      dyna_cloud->clear();
    }

    if (use_dynamic_sc_) {
      if (cloud_filtered->empty()) {
        ROS_WARN("Empty static cloud for keyframe %d, skipping keyframe", kf.id);
        continue;
      }
      kf.cloud = cloud_filtered;
      // 空动态点云也走动态接口，保持描述子、ring key和mask的定义一致。
      seq_relocalizer_.makeAndSaveScancontextAndKeysWithDynamic(*cloud_filtered, *dyna_cloud);
    } else {
      // 合并静态和动态点云，生成原始SC
      pcl::PointCloud<PointType>::Ptr combined_cloud(new pcl::PointCloud<PointType>(*cloud_filtered));
      *combined_cloud += *dyna_cloud;
      if (combined_cloud->empty()) {
        ROS_WARN("Empty combined cloud for keyframe %d, skipping keyframe", kf.id);
        continue;
      }
      kf.cloud = combined_cloud;
      seq_relocalizer_.makeAndSaveScancontextAndKeys(*combined_cloud);
    }

    keyframes_.push_back(kf);
  }
  file.close();
  ROS_INFO("Loaded %lu keyframes from map.", keyframes_.size());
}

bool SeqMapLocalization::alignClouds(pcl::PointCloud<PointType>::Ptr source,
                                   pcl::PointCloud<PointType>::Ptr target,
                                   Eigen::Matrix4f& transform) {
  const auto start = localization_runtime::Clock::now();
  pcl::IterativeClosestPoint<PointType, PointType> ndt;
  ndt.setTransformationEpsilon(trans_eps_);
  ndt.setEuclideanFitnessEpsilon(fitness_eps_);
  ndt.setMaximumIterations(max_iter_);
  ndt.setMaxCorrespondenceDistance(max_corr_dist_);
  // ndt.setResolution(ndt_resolution_);

  ndt.setInputSource(source);
  ndt.setInputTarget(target);

  pcl::PointCloud<PointType> output;
  // std::cout << "222" << std::endl;
  ndt.align(output);
  runtime_.addICP(localization_runtime::elapsedMs(start));
  if (ndt.hasConverged()) {
    transform = ndt.getFinalTransformation();
    return true;
  }
  return false;
}

void SeqMapLocalization::radarCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  if (use_dynamic_sc_) return;
  raw_points_buffer_ = msg;
  processPoints();
}

void SeqMapLocalization::staticPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  // 保存静态点云
  if (use_dynamic_sc_) {
    static_points_buffer_ = msg;
    processPoints();
  }
}

void SeqMapLocalization::dynamicPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  // 保存动态点云
  if (use_dynamic_sc_) {
    dynamic_points_buffer_ = msg;
    processPoints();
  }
}

void SeqMapLocalization::processPoints() {
  sensor_msgs::PointCloud2ConstPtr query_msg;
  pcl::PointCloud<PointType>::Ptr dynamic_cloud(new pcl::PointCloud<PointType>);
  if (use_dynamic_sc_) {
    if (!static_points_buffer_ || !dynamic_points_buffer_) {
      return;
    }
    
    // 配对前端的静态和动态点云，丢弃超出20ms的较旧消息。
    const double time_diff = (static_points_buffer_->header.stamp -
                              dynamic_points_buffer_->header.stamp).toSec();
    if (std::abs(time_diff) > 0.02) {
      ROS_WARN_THROTTLE(5, "Static and dynamic point clouds are not synchronized (time diff: %.3f s)", std::abs(time_diff));
      if (time_diff < 0) {
        static_points_buffer_.reset();
      } else {
        dynamic_points_buffer_.reset();
      }
      return;
    }

    query_msg = static_points_buffer_;
    const auto dynamic_msg = dynamic_points_buffer_;
    static_points_buffer_.reset();
    dynamic_points_buffer_.reset();
    pcl::fromROSMsg(*dynamic_msg, *dynamic_cloud);
  } else {
    if (!raw_points_buffer_) return;
    query_msg = raw_points_buffer_;
    raw_points_buffer_.reset();
  }

  // true: query_cloud为静态点云；false: query_cloud为原始完整点云。
  pcl::PointCloud<PointType>::Ptr query_cloud(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(*query_msg, *query_cloud);
  if (query_cloud->empty()) {
    ROS_WARN("Empty %s point cloud received", use_dynamic_sc_ ? "static" : "raw");
    return;
  }
  if (keyframes_.empty()) {
    ROS_WARN_THROTTLE(5, "No map keyframes loaded; skipping localization.");
    return;
  }

  // 与地图构建模式一致，将对应输入加入序列查询窗口。
  localization_runtime::Profiler::Frame timing(runtime_, query_msg->header.stamp.toSec());
  localization_runtime::StageTimer descriptor_timer(timing.sample.descriptor_ms);
  if (use_dynamic_sc_) {
    seq_relocalizer_.addFrameWithWindowMultiRes(*query_cloud, *dynamic_cloud);
  } else {
    seq_relocalizer_.addFrameWithWindowMultiRes(*query_cloud);
  }
  descriptor_timer.stop();
  timing.sample.kind = seq_relocalizer_.getCurrentWindowSize() < seq_relocalizer_.getFineWindowSize()
      ? "warmup" : "window_update";
  timing.sample.status = "buffered";

  static int frame_counter = 0;
  frame_counter++;
  if (frame_counter % 5 == 0 && seq_relocalizer_.getCurrentWindowSize() >= seq_relocalizer_.getFineWindowSize()) {
      timing.sample.kind = "query";
      timing.sample.status = "rejected";
      
      int best_frame_id = -1;
      float yaw_diff_rad = 0.0;
      std::vector<int> candidates;
      bool success = false;
      float score_conv;
      
      // seq_type:1. 调用粗到细序列匹配;  2. 调用粒子滤波序列匹配
      if (seq_type == 1){
        success = seq_relocalizer_.relocalizeCoarseToFine(best_frame_id, yaw_diff_rad, candidates);
      }
      else if (seq_type == 2){
        success = seq_relocalizer_.relocalizeCoarseToFineAligned(best_frame_id, yaw_diff_rad, candidates);
      }
      const auto& stages = seq_relocalizer_.getLastTiming();
      timing.sample.retrieval_ms = stages.retrieval_ms;
      timing.sample.voting_ms = stages.voting_ms;
      timing.sample.fine_ms = stages.fine_ms;
      // else if (seq_type == 3){
      //   success = seq_relocalizer_.relocalizeByConv(best_frame_id, yaw_diff_rad, score_conv);
      // }
      if (success && best_frame_id >= 0 && best_frame_id < keyframes_.size()) {
        timing.sample.status = "icp_failed";
        // 4. 使用匹配到的关键帧进行精配准
        Eigen::Matrix4f transform;
        if (alignClouds(query_cloud, keyframes_[best_frame_id].cloud, transform)) {
            // 计算最终
          Eigen::Matrix4f kf_pose = Eigen::Matrix4f::Identity();
          Eigen::Quaternionf kf_quat(keyframes_[best_frame_id].orientation.cast<float>());
          Eigen::Vector3f kf_trans(keyframes_[best_frame_id].position.cast<float>());
          kf_pose.block<3,3>(0,0) = kf_quat.toRotationMatrix();
          kf_pose.block<3,1>(0,3) = kf_trans;

          // 当前雷达在地图坐标系下的位姿 = 关键帧位姿 * 相对变换
          Eigen::Matrix4f world_pose = kf_pose * transform;
          Eigen::Vector3f translation = world_pose.block<3,1>(0,3);
          Eigen::Quaternionf rotation(world_pose.block<3,3>(0,0));

          // 发布Odometry消息
          nav_msgs::Odometry odom;
          odom.header.stamp = query_msg->header.stamp;  // 使用当前模式查询点云的时间戳
          odom.header.frame_id = "map";
          odom.child_frame_id = "radar";
          odom.pose.pose.position.x = translation.x();
          odom.pose.pose.position.y = translation.y();
          odom.pose.pose.position.z = translation.z();
          odom.pose.pose.orientation.x = rotation.x();
          odom.pose.pose.orientation.y = rotation.y();
          odom.pose.pose.orientation.z = rotation.z();
          odom.pose.pose.orientation.w = rotation.w();
          odom.pose.covariance[0] = seq_relocalizer_.getLastMatchScore();
          odom.pose.covariance[1] = static_cast<double>(keyframes_[best_frame_id].id);  // 原始地图文件中的关键帧ID
          pub_odom_.publish(odom);
          odom.header.frame_id = "camera_init";
          pub_odom__camera_init.publish(odom);

          // 发布TF变换
          tf::Transform tf_transform;
          tf_transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
          tf_transform.setRotation(tf::Quaternion(rotation.x(), rotation.y(), rotation.z(), rotation.w()));
          tf_broadcaster_.sendTransform(tf::StampedTransform(tf_transform, 
                                                            query_msg->header.stamp, 
                                                            "map", "radar"));

          // 发布回环序号
          std_msgs::Int32 loop_idx_msg;
          loop_idx_msg.data = keyframes_[best_frame_id].id;
          pub_loop_idx_.publish(loop_idx_msg);
          timing.sample.status = "published";
          timing.finish();  // Exclude GT evaluation below; early failures use the scope guard.

          ROS_DEBUG("Matched keyframe %d , pose: (%.2f, %.2f, %.2f)",
          keyframes_[best_frame_id].id, translation.x(), translation.y(), translation.z());
          if (has_gt_) {

            // 时间对齐检查（非常重要！）
            double time_diff = fabs(query_msg->header.stamp.toSec() -
                                    current_gt_time_.toSec());

            if (time_diff > 0.3) {  // 300ms
              ROS_WARN_THROTTLE(5, "GT not synchronized (dt=%.3f)", time_diff);
              return;
            }

            total_queries_++;

            // 当前估计位姿
            Eigen::Vector3d est_pos(
                translation.x(),
                translation.y(),
                translation.z()
            );

            // 匹配到的关键帧位置（place recognition）
            Eigen::Vector3d kf_pos = keyframes_[best_frame_id].position;

            // 误差
            double err_est = (est_pos - current_gt_pos_).norm();
            double err_kf  = (kf_pos  - current_gt_pos_).norm();

            // Recall定义（建议用KF版本）
            if (err_kf < recall_threshold_) {
              correct_queries_++;
            }

            double recall = static_cast<double>(correct_queries_) / total_queries_;

            ROS_INFO("[Recall] %.4f | KF_err: %.2f | Est_err: %.2f | dt: %.3f",
                    recall, err_kf, err_est, time_diff);

            sum_sq_error_ += err_est * err_est;
            double rmse = sqrt(sum_sq_error_ / total_queries_);

            ROS_INFO("[Metrics] Recall: %.4f | RMSE: %.3f",
                    recall, rmse);
            log_file << "[Metrics] Recall: "<< recall  <<  " RMSE: "<< rmse << std::endl;
          }
        }
      }
      else 
      {
        ROS_WARN("Sequence relocalization failed");
      }
  }
  
  
  // 缓冲区已在处理开始时消费，提前返回也不会重复加入旧帧。
}

void SeqMapLocalization::gtPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  current_gt_pos_ = Eigen::Vector3d(
      msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z
  );
  ROS_INFO("gt pose received!");
  current_gt_time_ = msg->header.stamp;
  has_gt_ = true;
}

void SeqMapLocalization::run() {
  ros::spin();
}

int main(int argc, char** argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  ros::init(argc, argv, "seq_place_recognition");
  SeqMapLocalization loc;
  loc.run();
  return 0;
}
