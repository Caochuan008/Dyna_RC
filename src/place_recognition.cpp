#include "dyna_rc/place_recognition.h"
#include <cmath>

MapLocalization::MapLocalization() {
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
  nh_.param("radar_stat_topic", radar_stat_topic_, std::string("/radar_points_inlier"));
  nh_.param("radar_dyna_topic", radar_dyna_topic_, std::string("/radar_points_outlier"));
  nh_.param("use_dynamic_sc", use_dynamic_sc_, false);  // 新增：是否使用动态SC
  nh_.param("recall_threshold", recall_threshold_, 150.0);
  nh_.param("z_min", z_min_, -2.0);
  nh_.param("z_max", z_max_, 50.0); 

  nh_.param("gt_topic", gt_topic_, std::string("/gt_pose"));
  nh_.param("sc_channel", sc_channel_, 1);  //1:z  2:intensity

  // 配置 SCManager
  sc_manager_.setSCdistThres(sc_dist_thres_);
  sc_manager_.setMaximumRadius(sc_max_radius_);
  sc_manager_.setScChannel(sc_channel_);

  // 加载地图
  loadMap();
  runtime_.configure(nh_, use_dynamic_sc_ ? "RC" : "SC-z", keyframes_.size());

  // 两种模式都订阅前端分离后的两路点云：普通SC合并使用，动态SC分开使用。
  sub_radar_stat_ = nh_.subscribe(radar_stat_topic_, 1, &MapLocalization::staticPointsCallback, this);
  sub_radar_dyna_ = nh_.subscribe(radar_dyna_topic_, 1, &MapLocalization::dynamicPointsCallback, this);
  ROS_INFO("%s SC: static=%s, dynamic=%s", use_dynamic_sc_ ? "Dynamic" : "Combined",
           radar_stat_topic_.c_str(), radar_dyna_topic_.c_str());
  sub_gt_pose_ = nh_.subscribe(gt_topic_, 10, &MapLocalization::gtPoseCallback, this);
  // 发布定位结果和回环序号
  pub_odom_ = nh_.advertise<nav_msgs::Odometry>("/localization_odom", 10);
  pub_loop_idx_ = nh_.advertise<std_msgs::Int32>("/loop_closure_index", 10);
  pub_candidates_ = nh_.advertise<geometry_msgs::PoseArray>("/localization/candidates", 10);
  pub_candidate_scores_ = nh_.advertise<std_msgs::Float32MultiArray>("/localization/candidate_scores", 10);
  pub_combined_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/radar_points_withdynamic", 10);


  if (use_dynamic_sc_) {
    ROS_INFO("MapLocalization initialized with dynamic SC, loaded %lu keyframes.", keyframes_.size());
  } else {
    ROS_INFO("MapLocalization initialized with original SC, loaded %lu keyframes.", keyframes_.size());
  }

  std::string log_path;
  nh_.param("metrics_log_file", log_path, std::string());
  if (!log_path.empty()) {
    log_file.open(log_path, std::ios::out | std::ios::app);
    if (!log_file.is_open()) ROS_WARN("Cannot open metrics_log_file: %s", log_path.c_str());
  }
}

void MapLocalization::loadMap() {
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
      sc_manager_.makeAndSaveScancontextAndKeysWithDynamic(*cloud_filtered, *dyna_cloud);
    } else {
      // 合并静态和动态点云，生成原始SC
      pcl::PointCloud<PointType>::Ptr combined_cloud(new pcl::PointCloud<PointType>(*cloud_filtered));
      *combined_cloud += *dyna_cloud;
      if (combined_cloud->empty()) {
        ROS_WARN("Empty combined cloud for keyframe %d, skipping keyframe", kf.id);
        continue;
      }
      kf.cloud = combined_cloud;
      sc_manager_.makeAndSaveScancontextAndKeys(*combined_cloud);
    }

    keyframes_.push_back(kf);
  }
  file.close();
  ROS_INFO("Loaded %lu keyframes from map.", keyframes_.size());
}

bool MapLocalization::alignClouds(pcl::PointCloud<PointType>::Ptr source,
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

void MapLocalization::staticPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  // 保存静态点云
  static_points_buffer_ = msg;
  processPoints();
}

void MapLocalization::dynamicPointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  // 保存动态点云
  dynamic_points_buffer_ = msg;
  processPoints();
}

void MapLocalization::processPoints() {
  if (!static_points_buffer_ || !dynamic_points_buffer_) {
    return;
  }

  // 两种模式都配对前端的静态和动态点云，丢弃超出20ms的较旧消息。
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

  const auto query_msg = static_points_buffer_;
  const auto dynamic_msg = dynamic_points_buffer_;
  static_points_buffer_.reset();
  dynamic_points_buffer_.reset();

  pcl::PointCloud<PointType>::Ptr static_cloud(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr dynamic_cloud(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(*query_msg, *static_cloud);
  pcl::fromROSMsg(*dynamic_msg, *dynamic_cloud);

  // 普通SC提前合并XYZI点云；动态SC仍使用分开的静态和动态点云。
  pcl::PointCloud<PointType>::Ptr combined_cloud;
  pcl::PointCloud<PointType>::Ptr query_cloud = static_cloud;
  if (!use_dynamic_sc_) {
    combined_cloud.reset(new pcl::PointCloud<PointType>(*static_cloud));
    *combined_cloud += *dynamic_cloud;
    query_cloud = combined_cloud;
  }
  if (query_cloud->empty()) {
    ROS_WARN("Empty %s point cloud received", use_dynamic_sc_ ? "static" : "combined");
    return;
  }
  if (keyframes_.empty()) {
    ROS_WARN_THROTTLE(5, "No map keyframes loaded; skipping localization.");
    return;
  }

  // 描述子构建和检索接口成对选择，与loadMap中的模式保持一致。
  localization_runtime::Profiler::Frame timing(runtime_, query_msg->header.stamp.toSec());
  localization_runtime::StageTimer stage(timing.sample.descriptor_ms);
  std::pair<int, float> detectResult;
  if (use_dynamic_sc_) {
    sc_manager_.makeAndSaveScancontextAndKeysWithDynamic(*static_cloud, *dynamic_cloud);
    stage.next(timing.sample.retrieval_ms);
    detectResult = sc_manager_.detectLoopClosureIDWithDynamic();
  } else {
    sc_manager_.makeAndSaveScancontextAndKeys(*combined_cloud);
    stage.next(timing.sample.retrieval_ms);
    detectResult = sc_manager_.detectLoopClosureID();
  }
  stage.stop();
  // 两种模式均移除临时查询，避免实时帧污染地图描述子库。
  sc_manager_.CancelScancontextAndKeysWithDynamic();

  auto top_candidates = sc_manager_.getTopCandidates();

  // stage.next(timing.sample.candidates_ms);
  // publishCandidatePoses(
  //     top_candidates,
  //     query_cloud,
  //     query_msg->header.stamp
  // );
  stage.stop();
  timing.sample.candidate_icp_ms = timing.sample.icp_ms;
  timing.sample.candidate_icp_calls = timing.sample.icp_calls;

  int best_idx = detectResult.first;   // 关键帧索引
  float score = detectResult.second;

  if (best_idx < 0 || best_idx >= static_cast<int>(keyframes_.size())) {
    ROS_WARN_THROTTLE(5, "No matching keyframe found for current scan (score=%.3f).", score);
    return;
  }

  // 使用ICP精确配准
  timing.sample.status = "icp_failed";
  Eigen::Matrix4f transform;
  if (!alignClouds(query_cloud, keyframes_[best_idx].cloud, transform)) {
    ROS_WARN_THROTTLE(5, "ICP alignment failed for candidate %d.", keyframes_[best_idx].id);
    return;
  }

  // 构造关键帧的位姿矩阵
  Eigen::Matrix4f kf_pose = Eigen::Matrix4f::Identity();
  Eigen::Quaternionf kf_quat(keyframes_[best_idx].orientation.cast<float>());
  Eigen::Vector3f kf_trans(keyframes_[best_idx].position.cast<float>());
  kf_pose.block<3,3>(0,0) = kf_quat.toRotationMatrix();
  kf_pose.block<3,1>(0,3) = kf_trans;

  // 当前雷达在地图坐标系下的位姿 = 关键帧位姿 * 相对变换
  Eigen::Matrix4f world_pose = kf_pose * transform;
  Eigen::Vector3f translation = world_pose.block<3,1>(0,3);
  Eigen::Quaternionf rotation(world_pose.block<3,3>(0,0));

  // 发布Odometry消息
  nav_msgs::Odometry odom;
  odom.header.stamp = query_msg->header.stamp;
  odom.header.frame_id = "map";
  odom.child_frame_id = "radar";
  odom.pose.pose.position.x = translation.x();
  odom.pose.pose.position.y = translation.y();
  odom.pose.pose.position.z = translation.z();
  odom.pose.pose.orientation.x = rotation.x();
  odom.pose.pose.orientation.y = rotation.y();
  odom.pose.pose.orientation.z = rotation.z();
  odom.pose.pose.orientation.w = rotation.w();
  odom.pose.covariance[0] = score;
  odom.pose.covariance[1] = static_cast<double>(keyframes_[best_idx].id);
  pub_odom_.publish(odom);

  // 发布TF变换
  tf::Transform tf_transform;
  tf_transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
  tf_transform.setRotation(tf::Quaternion(rotation.x(), rotation.y(), rotation.z(), rotation.w()));
  tf_broadcaster_.sendTransform(tf::StampedTransform(tf_transform, 
                                                      query_msg->header.stamp, 
                                                     "map", "radar"));

  // 发布回环序号
  std_msgs::Int32 loop_idx_msg;
  loop_idx_msg.data = keyframes_[best_idx].id;
  pub_loop_idx_.publish(loop_idx_msg);
  timing.sample.status = "published";
  timing.finish();  // Exclude GT evaluation and visualization below.

  ROS_DEBUG("Matched keyframe %d (score=%.3f), pose: (%.2f, %.2f, %.2f)",
            keyframes_[best_idx].id, score, translation.x(), translation.y(), translation.z());

    // ================= Recall Evaluation =================
  if (has_gt_) {

    // 时间对齐检查（非常重要！）
    double time_diff = fabs(query_msg->header.stamp.toSec() -
                            current_gt_time_.toSec());
    // std::cout<< "start recall,  time_diff is" << time_diff <<std::endl;
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
    Eigen::Vector3d kf_pos = keyframes_[best_idx].position;

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
  
  // ===== 创建带动态标签的发布点云（SC使用上面的XYZI点云） =====
  pcl::PointCloud<PointXYZIDynamic>::Ptr labeled_cloud(new pcl::PointCloud<PointXYZIDynamic>);
  labeled_cloud->header = static_cloud->header;
  labeled_cloud->header.frame_id = "radar";  // 根据实际情况调整 frame_id
  
  // 两种模式都从分开的输入添加静态点，避免重复加入动态点。
  if (!static_cloud->empty()) {
      for (const auto& pt : static_cloud->points) {
          PointXYZIDynamic p;
          p.x = pt.x;
          p.y = pt.y;
          p.z = pt.z;
          p.intensity = pt.intensity;
          p.dynamic = 0;  // 静态点标记为0
          labeled_cloud->push_back(p);
      }
  }
  
  // 添加动态点云（dynamic = 1）
  if (!dynamic_cloud->empty()) {
      for (const auto& pt : dynamic_cloud->points) {
          PointXYZIDynamic p;
          p.x = pt.x;
          p.y = pt.y;
          p.z = pt.z;
          p.intensity = pt.intensity;
          p.dynamic = 1;  // 动态点标记为1
          labeled_cloud->push_back(p);
      }
  }
  
  // 转换为 ROS 消息并发布
  if (!labeled_cloud->empty()) {
      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(*labeled_cloud, cloud_msg);
      cloud_msg.header.stamp = query_msg->header.stamp;
      cloud_msg.header.frame_id = "radar";  // 与上面的 frame_id 保持一致
      pub_combined_cloud_.publish(cloud_msg);
      
      ROS_DEBUG("Published combined cloud with %lu points (static: %lu, dynamic: %lu)",
                labeled_cloud->size(),
                static_cloud->size(),
                dynamic_cloud->size());
  }

  // 输入缓冲区已在处理开始时消费，提前返回也不会留下旧帧。
}

void MapLocalization::publishCandidatePoses(
    const std::vector<std::pair<int, float>>& candidates,
    pcl::PointCloud<PointType>::Ptr current_cloud,
    const ros::Time& stamp) {

    geometry_msgs::PoseArray pose_array;
    pose_array.header.stamp = stamp;
    pose_array.header.frame_id = "map";
    std_msgs::Float32MultiArray score_array;

    for (const auto& cand : candidates) {
        int idx = cand.first;
        float similiarity_knn = cand.second;
        if (idx < 0 || idx >= static_cast<int>(keyframes_.size()))
            continue;

        // 1. NDT 配准
        Eigen::Matrix4f relative_tf = Eigen::Matrix4f::Identity();
        bool aligned = alignClouds(current_cloud,
                                   keyframes_[idx].cloud,
                                   relative_tf);
        if (!aligned)
            continue;

        // 2. 候选关键帧的世界位姿
        Eigen::Matrix4f kf_pose = Eigen::Matrix4f::Identity();
        Eigen::Quaternionf q(keyframes_[idx].orientation.cast<float>());
        Eigen::Vector3f t(keyframes_[idx].position.cast<float>());

        kf_pose.block<3,3>(0,0) = q.toRotationMatrix();
        kf_pose.block<3,1>(0,3) = t;

        // 3. 当前雷达的世界位姿
        Eigen::Matrix4f world_pose = kf_pose * relative_tf;

        Eigen::Vector3f final_t = world_pose.block<3,1>(0,3);
        Eigen::Quaternionf final_q(world_pose.block<3,3>(0,0));

        // 4. 构造 Pose
        geometry_msgs::Pose pose;
        pose.position.x = final_t.x();
        pose.position.y = final_t.y();
        pose.position.z = final_t.z();
        pose.orientation.x = final_q.x();
        pose.orientation.y = final_q.y();
        pose.orientation.z = final_q.z();
        pose.orientation.w = final_q.w();

        pose_array.poses.push_back(pose);
        score_array.data.push_back(similiarity_knn);
    }

    pub_candidates_.publish(pose_array);
    pub_candidate_scores_.publish(score_array);
}

void MapLocalization::gtPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  current_gt_pos_ = Eigen::Vector3d(
      msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z
  );
  ROS_INFO("gt pose received!");
  current_gt_time_ = msg->header.stamp;
  has_gt_ = true;
}

void MapLocalization::run() {
  ros::spin();
}

int main(int argc, char** argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  ros::init(argc, argv, "place_recognition");
  MapLocalization loc;
  loc.run();
  return 0;
}
