#include "dyna_rc/radar_ego_motion.h"
#include "dyna_rc/runtime_timing.h"

namespace {
class RadarTimer {
 public:
  double toc_silent() const { return localization_runtime::elapsedMs(start_); }
  void toc(const char* name) const { ROS_INFO("%s: %.3f ms", name, toc_silent()); }
 private:
  localization_runtime::Clock::time_point start_ = localization_runtime::Clock::now();
};
constexpr double kPi = 3.14159265358979323846;
}

RadarEgoMotion::RadarEgoMotion() {
  pub_ego_vel_ =
      nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("/ego_vel", 10);
  pub_radar_vel_ =
      nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("/radar_vel", 10);
  pub_outlier_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/radar_points_outlier", 100);
  pub_inlier_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/radar_points_inlier", 100);
  pub_inlier_2d_ =
      nh_.advertise<sensor_msgs::PointCloud2>("/radar_points_inlier_2d", 100);

  param("radar_topic", rad_topic, std::string("/radar_v_points"));
  param("debug_mode", debug, false);
  param("use_rcs", use_rcs, true);
  param("tictoc", tictoc, false);
  param("min_height", min_height, -10.0);
  param("max_height", max_height, 50.0);
  param("min_intensity", min_intensity, 0.0);
  param("max_scan_range", max_scan_range, 250.0);
  param("min_scan_range", min_scan_range, 0.0);
  param("n_dof", n_dof_, 3);
  param("input_leaf_size", input_leaf_size_, 0.3);  // Original source does not voxel-filter.
  param("ransac_threshold", ransac_threshold, 0.95);
  param("t_x", t_x, 0.57);
  param("t_y", t_y, 0.0);
  param("t_z", t_z, 1.12);
  param("mounting_angle_x", mounting_angle_x, 3.1415926);
  param("mounting_angle_y", mounting_angle_y, 0.0);
  param("mounting_angle_z", mounting_angle_z, 0.0);
  param("radar_type", radar_type_, 1);
  param("velocity_accuracy", velocity_accuracy, 0.02);
  param("unstable_filter", unstable_filter, true);
  param("ground_filter", ground_filter_enabled_, true);
  param("unstable_radius", unstable_radius_, 1.0);
  param("unstable_max_dt", unstable_max_dt_, 1.0);
  for (double v : {min_height, max_height, min_intensity, max_scan_range, min_scan_range,
                  t_x, t_y, t_z, mounting_angle_x, mounting_angle_y, mounting_angle_z,
                  ransac_threshold, velocity_accuracy, unstable_radius_, unstable_max_dt_, input_leaf_size_})
    if (!std::isfinite(v)) throw std::runtime_error("Radar preprocessing parameters must be finite");
  if ((n_dof_ != 2 && n_dof_ != 3 && n_dof_ != 6) || (radar_type_ != 1 && radar_type_ != 2) ||
      max_height <= min_height || min_scan_range < 0 || max_scan_range <= min_scan_range ||
      ransac_threshold <= 0 || velocity_accuracy <= 0 || unstable_radius_ <= 0 ||
      unstable_max_dt_ <= 0 || t_z < 0)
    throw std::runtime_error("Invalid radar preprocessing parameters");

  ransac_iter_ = calc_ransac_iter(n_dof_);
  total_frames_ = 0;
  total_ego_vel_time_ = 0.0;
  std::string log_path;
  param("timing_log_file", log_path, std::string());
  if (!log_path.empty()) {
    log_file.open(log_path, std::ios::out | std::ios::app);
    if (!log_file) ROS_WARN("Cannot open optional timing log: %s", log_path.c_str());
  }
  sub_scan_ = nh_.subscribe<sensor_msgs::PointCloud2>(rad_topic, 100, &RadarEgoMotion::ScanCallback, this);
  ROS_INFO("Radar preprocessing: %s -> inlier/outlier; intensity=%s, ground=%d, unstable=%d, RANSAC=%.3f",
           rad_topic.c_str(), use_rcs ? "rcs" : "intensity", ground_filter_enabled_, unstable_filter, ransac_threshold);
}

bool RadarEgoMotion::msg_to_pcl(const sensor_msgs::PointCloud2 &scan_msg,
                pcl::PointCloud<RadarPointType>::Ptr scan_pcl) {
  scan_pcl->clear();
  if (!scan_msg.width || !scan_msg.height) return true;
  std::set<std::string> fields;
  std::string fields_str;

  for (const auto &field : scan_msg.fields) {
    fields.emplace(field.name);
    fields_str += field.name + ", ";
  }
  for (const char* name : {"x", "y", "z", "intensity", "range", "azimuth", "elevation", "velocity", "rcs"}) {
    if (!fields.count(name)) {
      ROS_ERROR_THROTTLE(5, "Radar input is missing field '%s'", name);
      return false;
    }
  }

  if (fields.find("snr") != fields.end() &&
      fields.find("rcs") != fields.end() &&
      fields.find("detectionconfidence") != fields.end() &&
      fields.find("velocity") != fields.end()) {
    ROS_INFO_ONCE("Detected ZF radar format points!");
    pcl::PointCloud<ZFRadarPointType> zf_scan;
    pcl::fromROSMsg(scan_msg, zf_scan);

    scan_pcl->clear();
    scan_pcl->reserve(zf_scan.size());
    // 恢复雷达坐标
    if (radar_type_ == 1)
    {
      for (int i = 0; i< zf_scan.size() ; i++) {
        zf_scan.points[i].y = -zf_scan.points[i].y;  // Y 坐标取反
        zf_scan.points[i].z = -zf_scan.points[i].z;  // Z 坐标取反
      }
    }
    Eigen::Matrix3d R_y;
    R_y << cos(mounting_angle_y), 0.0, -sin(mounting_angle_y),
            0.0, 1.0, 0.0,
            sin(mounting_angle_y), 0.0, cos(mounting_angle_y);

    for (const auto &p : zf_scan) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
          !std::isfinite(p.intensity) || !std::isfinite(p.rcs) || !std::isfinite(p.range) ||
          !std::isfinite(p.azimuth) || !std::isfinite(p.elevation) || !std::isfinite(p.velocity)) continue;
      if (p.z < min_height) continue;
      if (p.z > max_height) continue;
      if (p.intensity < min_intensity) continue;
      if (p.range > max_scan_range) continue;
      if (p.range < min_scan_range) continue;
      Eigen::Vector3d point_vec(p.x, p.y, p.z);
      Eigen::Vector3d rotated_vec = R_y * point_vec;

      RadarPointType radarpoint{};
      radarpoint.data[3] = 1.0f;
      radarpoint.x          = rotated_vec(0);
      radarpoint.y          = rotated_vec(1);
      radarpoint.z          = rotated_vec(2);
      if (use_rcs){
        radarpoint.intensity  = p.rcs; // ::pow(10, p.rcs/10);
      }
      else{
        radarpoint.intensity  = p.intensity; // ::pow(10, p.rcs/10);
      }
      radarpoint.range      = p.range;
      radarpoint.azimuth    = p.azimuth;
      radarpoint.elevation  = p.elevation;
      radarpoint.velocity   = p.velocity;
      radarpoint.rcs        = p.rcs;
      radarpoint.flag       = 0;
      scan_pcl->push_back(radarpoint);
    }

    return true;
  } else {
    ROS_ERROR_STREAM_ONCE("Unsupported point format with fields: "
                          << fields_str.substr(0, fields_str.size() - 2));
    return false;
  }
}

std::vector<int> RadarEgoMotion::Ransac(const Eigen::VectorXf &v_doppler,
                        const Eigen::MatrixXf &M_mat, double threshold) {
  if (M_mat.rows() < 3 || M_mat.cols() < 2 || M_mat.rows() != v_doppler.size()) return {};
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  cloud->width = M_mat.rows();
  cloud->height = 1;
  cloud->is_dense = true;
  cloud->points.resize(cloud->width * cloud->height);

  for (int i = 0; i < cloud->points.size(); ++i) {
    cloud->points[i].x = M_mat(i, 0);
    cloud->points[i].y = M_mat(i, 1);
    cloud->points[i].z = v_doppler[i];
  }

  std::vector<int> inlier_idx;
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));

  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
  ransac.setMaxIterations(200);
  ransac.setDistanceThreshold(threshold);
  ransac.computeModel();
  ransac.getInliers(inlier_idx);

  return inlier_idx;
}

std::vector<int> RadarEgoMotion::Ransac(const Eigen::VectorXf &v_doppler,
                                   const Eigen::MatrixXf &M_mat,
                                   int N_ransac_points, int ransac_iter,
                                   double threshold) {
  std::vector<int> inlier_idx_best;
  if (v_doppler.size() == 0 || M_mat.rows() != v_doppler.size()) return inlier_idx_best;

  int inlier_threshold = v_doppler.size() * 0.3;
  double v_sum = 0;
  for (int i = 0; i < v_doppler.size(); i++) {
    if (std::fabs(v_doppler[i]) < 0.05) {
      inlier_idx_best.emplace_back(i);
      v_sum += v_doppler[i];
    }
  }
  v_sum /= static_cast<float>(v_doppler.size());
  if (inlier_idx_best.size() > inlier_threshold) {
    if (debug) {
      std::cout << "Static Ego-Vehicle!" << inlier_idx_best.size() << ", " << v_doppler.size() <<", v_average:" << v_sum << std::endl;
    }
    return inlier_idx_best;
  }
  if (debug) {
    std::cout << "Dynamic Ego-Vehicle!" << std::endl;
  }

  std::vector<int> idx(v_doppler.rows());
  for (int k = 0; k < v_doppler.rows(); k++) {
    idx[k] = k;
  }
  std::random_device rd;
  std::mt19937 g(rd());

  if (v_doppler.rows() >= N_ransac_points) {
    for (int k = 0, kk = 0; k < 200 && kk < ransac_iter; ++k) {
      std::shuffle(idx.begin(), idx.end(), g);
      Eigen::VectorXf VD_samples(N_ransac_points);
      Eigen::MatrixXf M_samples(N_ransac_points, M_mat.cols());

      // 随机取n个点
      for (int i = 0; i < N_ransac_points; ++i) {
        VD_samples.row(i) = v_doppler.row(idx.at(i));
        M_samples.row(i) = M_mat.row(idx.at(i));
      }

      Eigen::JacobiSVD<Eigen::MatrixXf> svd(M_samples);
      double cond = svd.singularValues()(0) /
                    svd.singularValues()(svd.singularValues().size() - 1);
      //   std::cout << "conditional num: " << cond << std::endl;
      if (std::fabs(cond) < 1.0e3 || k > 100) {
        Eigen::VectorXf vel_radar_sample =
            M_samples.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(VD_samples);
        //   std::cout << vel_radar_sample << std::endl;
        const Eigen::VectorXf err =
            (v_doppler - M_mat * vel_radar_sample).array().abs();
        std::vector<int> inlier_idx;

        for (int j = 0; j < err.rows(); ++j) {
          if (err(j) < threshold) {
            inlier_idx.emplace_back(j);
          }
        }
        if (inlier_idx.size() > inlier_idx_best.size()) {
          inlier_idx_best = inlier_idx;
        }
        kk++;
      }
    }
  } else {
    ROS_WARN("Ransac outlier removal failed!");
    inlier_idx_best = idx;
  }
  return inlier_idx_best;
}

void RadarEgoMotion::doppler_ego_motion(
    const pcl::PointCloud<RadarPointType> &scan_pcl,
    Eigen::VectorXf *v_doppler, Eigen::MatrixXf *M_mat) {
  int scan_size = scan_pcl.points.size();
  v_doppler->resize(scan_size);
  if (n_dof_ == 2) {
    M_mat->resize(scan_size, 2);
  } else {
    M_mat->resize(scan_size, 3);
  }

  for (int i = 0; i < scan_size; ++i) {
    float theta = scan_pcl.points[i].azimuth;
    float beta = scan_pcl.points[i].elevation;
    // M is the relation between doppler and v_radar in radar sensor frame
    if (n_dof_ == 2) {
      M_mat->row(i) << cos(beta)*cos(theta), cos(beta)*sin(theta);
    } else {
      M_mat->row(i) << cos(beta)*cos(theta), cos(beta)*sin(theta), sin(beta);
    }
    (*v_doppler)[i] = -scan_pcl.points[i].velocity;
  }
}

void RadarEgoMotion::ground_filter(const pcl::PointCloud<RadarPointType> &radar_scan,
    pcl::PointCloud<RadarPointType>::Ptr scan_filtered, double sensor_height) {
  scan_filtered->clear();
  if (radar_scan.empty()) return;
  pcl::PointCloud<RadarPointType>::Ptr range_filtered(new pcl::PointCloud<RadarPointType>);
  pcl::PassThrough<RadarPointType> range_pass;
  range_pass.setInputCloud(radar_scan.makeShared());
  range_pass.setFilterFieldName("range");
  range_pass.setFilterLimits(0.0, 50.0);
  range_pass.filter(*range_filtered);

  pcl::PointCloud<RadarPointType>::Ptr height_filtered(new pcl::PointCloud<RadarPointType>);
  pcl::PassThrough<RadarPointType> height_pass;
  height_pass.setInputCloud(range_filtered);
  height_pass.setFilterFieldName("z");
  height_pass.setFilterLimits(-(sensor_height+0.25), -(sensor_height-0.25));
  height_pass.filter(*height_filtered);

  if(height_filtered->size() < 6) {
    height_pass.setInputCloud(radar_scan.makeShared());
    height_pass.setFilterFieldName("z");
    height_pass.setFilterLimits(-(sensor_height+0.3), max_height);
    height_pass.filter(*scan_filtered);
  } else {
    pcl::NormalEstimation<RadarPointType, pcl::Normal> ne;
    pcl::search::KdTree<RadarPointType>::Ptr tree(new pcl::search::KdTree<RadarPointType>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setInputCloud(height_filtered);
    ne.setSearchMethod(tree);
    ne.setKSearch(5);
    ne.setViewPoint(0.0, 0.0, sensor_height);
    ne.compute(*normals);

    pcl::PointCloud<RadarPointType>::Ptr normal_filtered(new pcl::PointCloud<RadarPointType>);
    normal_filtered->reserve(height_filtered->size());
    for(int i = 0; i < height_filtered->size(); i++) {
      float dot = normals->at(i).getNormalVector3fMap().normalized().dot(Eigen::Vector3f::UnitZ());
      if(std::isfinite(dot) && std::abs(dot) > std::cos(20.0 * kPi / 180.0)) {
        normal_filtered->push_back(height_filtered->at(i));
      }
    }

    if(normal_filtered->size() < 5) {
      std::cout << "too few points after normal filter: " << normal_filtered->size() << std::endl;
      height_pass.setInputCloud(radar_scan.makeShared());
      height_pass.setFilterFieldName("z");
      height_pass.setFilterLimits(-(sensor_height+0.2), max_height);
      height_pass.filter(*scan_filtered);
    } else {
      pcl::SampleConsensusModelPlane<RadarPointType>::Ptr model_p(new pcl::SampleConsensusModelPlane<RadarPointType>(normal_filtered));
      pcl::RandomSampleConsensus<RadarPointType> ransac(model_p);
      ransac.setMaxIterations(100);
      ransac.setDistanceThreshold(0.05);
      ransac.computeModel();

      pcl::PointIndices::Ptr ground_inlier(new pcl::PointIndices);
      ransac.getInliers(ground_inlier->indices);

      if(ground_inlier->indices.size() < 5) {
        std::cout << "too few points after ransac: " << ground_inlier->indices.size() << std::endl;
        height_pass.setInputCloud(radar_scan.makeShared());
        height_pass.setFilterFieldName("z");
        height_pass.setFilterLimits(-(sensor_height+0.2), max_height);
        height_pass.filter(*scan_filtered);
      } else {
        Eigen::VectorXf coeffs;
        ransac.getModelCoefficients(coeffs);
        if (coeffs.size() != 4) {
          height_pass.setInputCloud(radar_scan.makeShared());
          height_pass.setFilterLimits(-(sensor_height+0.2), max_height);
          height_pass.filter(*scan_filtered);
          return;
        }

        Eigen::Vector4f reference = Eigen::Vector4f::UnitZ();   // 0 0 1 0
        double dot = coeffs.head<3>().dot(reference.head<3>());
        if (debug) {
          std::cout << "plane coeffs: " << coeffs.transpose() << std::endl;
        }
        if(!coeffs.allFinite() || std::abs(dot) < std::cos(20.0 * kPi / 180.0)) {
          if (debug) std::cout << "Rejecting non-horizontal ground plane" << std::endl;
          height_pass.setInputCloud(radar_scan.makeShared());
          height_pass.setFilterFieldName("z");
          height_pass.setFilterLimits(-(sensor_height+0.2), max_height);
          height_pass.filter(*scan_filtered);
        } else {
          if(coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
            coeffs *= -1.0f;
          }

          pcl::PointCloud<RadarPointType>::Ptr inlier_cloud(new pcl::PointCloud<RadarPointType>);
          pcl::ExtractIndices<RadarPointType> extract_ground;
          extract_ground.setInputCloud(normal_filtered);
          extract_ground.setIndices(ground_inlier);
          extract_ground.filter(*inlier_cloud);

          pcl::PlaneClipper3D<RadarPointType> clipper(coeffs);
          pcl::PointIndices::Ptr indices(new pcl::PointIndices);
          clipper.clipPointCloud3D(radar_scan, indices->indices);

          pcl::ExtractIndices<RadarPointType> extract;
          extract.setInputCloud(radar_scan.makeShared());
          extract.setIndices(indices);
          extract.filter(*scan_filtered);
        }
      }
    }
  }
}

void RadarEgoMotion::unstable_point_detector(
    const geometry_msgs::TwistWithCovarianceStamped &radar_vel,
    pcl::PointCloud<RadarPointType> &inlier_points) {
  RadarTimer timer_kdtree;
  const double dt = (msg_header.stamp - last_radar_vel.header.stamp).toSec();
  if (inlier_points.empty() || last_inliers.empty() || dt <= 0 || dt > unstable_max_dt_) return;
  pcl::PointCloud<RadarPointType> now_inliers;
  pcl::copyPointCloud(inlier_points, now_inliers);

  Eigen::AngleAxisf rot_x((radar_vel.twist.twist.angular.x + last_radar_vel.twist.twist.angular.x) / 2 *
                              (msg_header.stamp - last_radar_vel.header.stamp).toSec(),
                          Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf rot_y((radar_vel.twist.twist.angular.y + last_radar_vel.twist.twist.angular.y) / 2 *
                              (msg_header.stamp - last_radar_vel.header.stamp).toSec(),
                          Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z((radar_vel.twist.twist.angular.z + last_radar_vel.twist.twist.angular.z) / 2 *
                              (msg_header.stamp - last_radar_vel.header.stamp).toSec(),
                          Eigen::Vector3f::UnitZ());
  Eigen::Translation3f translation(
      (radar_vel.twist.twist.linear.x + last_radar_vel.twist.twist.linear.x) / 2 *
          (msg_header.stamp - last_radar_vel.header.stamp).toSec(),
      (radar_vel.twist.twist.linear.y + last_radar_vel.twist.twist.linear.y) / 2 *
          (msg_header.stamp - last_radar_vel.header.stamp).toSec(),
      (radar_vel.twist.twist.linear.z + last_radar_vel.twist.twist.linear.z) / 2 *
          (msg_header.stamp - last_radar_vel.header.stamp).toSec());
  Eigen::Matrix4f tf_mat = (translation * rot_z * rot_y * rot_x).matrix();

  inlier_points.clear();

  pcl::PointCloud<RadarPointType> inliers_trans;
  pcl::transformPointCloud(last_inliers, inliers_trans, tf_mat.inverse());

  pcl::KdTreeFLANN<RadarPointType> kdtree;
  std::vector<int> search_idx;
  std::vector<float> search_dis;
  kdtree.setInputCloud(inliers_trans.makeShared());
  inlier_points.reserve(now_inliers.size());
  for (const auto & p : now_inliers) {
    search_idx.clear(); search_dis.clear();
    kdtree.radiusSearch(p, unstable_radius_, search_idx, search_dis);
    if (search_idx.size() > 0) {
      inlier_points.push_back(p);
    }
  }
  if (tictoc) {
    timer_kdtree.toc("unstable point detector");
  }
}

void RadarEgoMotion::ScanCallback(const sensor_msgs::PointCloud2ConstPtr &scan_msg) {
  int scan_size = 0;
  msg_header = scan_msg->header;
  pcl::PointCloud<RadarPointType>::Ptr radar_scan(new pcl::PointCloud<RadarPointType>);
  RadarTimer timer_radar_ego_motion;
  if (msg_to_pcl(*scan_msg, radar_scan)) {
    scan_size = radar_scan->points.size();
  } else {
    publishEmptyClouds();
    return;
  }
  if (scan_size == 0) { publishEmptyClouds(); return; }
  if (debug) {
    std::cout << "radar scan size: " << scan_size << std::endl;
  }

  // radar sensor pose in car frame

  pcl::PointCloud<RadarPointType>::Ptr scan_filtered(new pcl::PointCloud<RadarPointType>);
  if (ground_filter_enabled_) {
    ground_filter(*radar_scan, scan_filtered, t_z);
  } else *scan_filtered = *radar_scan;
  //   std::cout << i << "th src velocity:" << radar_scan->points[i].velocity << std::endl;
  //   std::cout << i << "th dsp velocity:" << scan_filtered->points[i].velocity << std::endl;

  if (scan_filtered->size() < static_cast<size_t>(n_dof_)) { publishEmptyClouds(); return; }
  Eigen::VectorXf v_doppler;
  Eigen::MatrixXf M_mat;
  doppler_ego_motion(*scan_filtered, &v_doppler, &M_mat);

  std::vector<int> inlier_idx;
  inlier_idx = Ransac(v_doppler, M_mat, n_dof_, ransac_iter_, ransac_threshold);
  if (debug) {
    std::cout << "inlier point size: " << inlier_idx.size() << std::endl;
  }

  int inlier_size = inlier_idx.size();
  if (inlier_size < n_dof_) { publishEmptyClouds(); return; }
  Eigen::MatrixXd inlier_M(inlier_size, M_mat.cols());
  Eigen::VectorXd inlier_doppler(inlier_size);
  for (int i = 0; i < inlier_size; ++i) {
    inlier_M.row(i) = M_mat.row(inlier_idx[i]).cast<double>();
    inlier_doppler(i) = static_cast<double>(v_doppler(inlier_idx[i]));
  }

  Eigen::VectorXd radar_motion =
      inlier_M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
          .solve(inlier_doppler);
  if (debug) {
    std::cout << radar_motion << std::endl;
  }

  // S is the relation between v_radar and ego-motion
  Eigen::MatrixXd S_mat;
  if (n_dof_ == 6) {
    // 3d 6 dof, ego-motion=[v_x; w_z; w_y; v_y; v_z; w_x]
    S_mat.resize(3, 6);
    S_mat <<  1.0, -t_y, t_z, 0.0, 0.0, 0.0,
              0.0, t_x, 0.0, 1.0, 0.0, -t_z,
              0.0, 0.0, -t_x, 0.0, 1.0, t_y;
  } else if (n_dof_ == 3) {
    // 3d 3 dof, ego-motion=[v_x; w_z; w_y]
    S_mat.resize(3, 3);
    S_mat << 1.0, -t_y, t_z,
             0.0,  t_x, 0.0,
             0.0, 0.0, -t_x;
  } else if (n_dof_ == 2) {
    // 2d 2 dof, ego-motion=[v_x; w_z]
    S_mat.resize(2, 2);
    S_mat << 1.0, -t_y,
             0.0, t_x;
  }
  Eigen::MatrixXd R_mat;
  if (n_dof_ == 2) {
  R_mat.resize(2, 2);
  R_mat << cos(mounting_angle_z), -sin(mounting_angle_z),
           sin(mounting_angle_z), cos(mounting_angle_z);
  } else {
  Eigen::AngleAxisd rot_x(mounting_angle_x, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd rot_y(mounting_angle_y, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_z(mounting_angle_z, Eigen::Vector3d::UnitZ());
  R_mat = (rot_z * rot_y * rot_x).matrix();
  }
  Eigen::MatrixXd new_inlier_M = inlier_M * R_mat.transpose() * S_mat;

  Eigen::VectorXd ego_motion =
      new_inlier_M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
          .solve(inlier_doppler);
  if (!radar_motion.allFinite() || !ego_motion.allFinite()) {
    ROS_WARN_THROTTLE(5, "Nonfinite Doppler velocity estimate; dropping this frame");
    publishEmptyClouds(); return;
  }

  if (debug) {
    std::cout << ego_motion << std::endl;
  }
  Eigen::MatrixXd H_skew;
    // 3d 6 dof, ego-motion=[v_x; w_z; w_y; v_y; v_z; w_x]
  H_skew.resize(3, 6);
  H_skew <<  1.0, 0.0, 0.0, 0.0, -t_y, t_z,
            0.0, 1.0, 0.0, -t_z, t_x, 0.0,
            0.0, 0.0, 1.0, t_y, 0.0, -t_x;
  Eigen::MatrixXd inlier_A = inlier_M * R_mat.transpose() * H_skew.topRows(R_mat.rows());
  Eigen::MatrixXd cov_mat = calc_cov(inlier_A, velocity_accuracy);

  geometry_msgs::TwistWithCovarianceStamped radar_vel;
  radar_vel.header.frame_id = "radar";
  radar_vel.header.seq = msg_header.seq;
  radar_vel.header.stamp = msg_header.stamp;
  radar_vel.twist.twist.linear.x = radar_motion(0);
  radar_vel.twist.twist.linear.y = radar_motion(1);
  radar_vel.twist.twist.linear.z = n_dof_ > 2 ? radar_motion(2) : 0.0;
  radar_vel.twist.twist.angular.x = n_dof_ > 5 ? ego_motion(5) : 0.0;
  radar_vel.twist.twist.angular.y = n_dof_ > 2 ? ego_motion(2) : 0.0;
  radar_vel.twist.twist.angular.z = ego_motion(1);

  for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c)
    radar_vel.twist.covariance[6*r+c] = cov_mat(r, c);

  geometry_msgs::TwistWithCovarianceStamped ego_vel;
  ego_vel.header.frame_id = "base_link";
  ego_vel.header.seq = msg_header.seq;
  ego_vel.header.stamp = msg_header.stamp;
  ego_vel.twist.twist.linear.x = ego_motion(0);
  ego_vel.twist.twist.angular.z = ego_motion(1);
  ego_vel.twist.twist.angular.y = n_dof_ > 2 ? ego_motion(2) : 0.0;
  ego_vel.twist.twist.linear.y = n_dof_ > 3 ? ego_motion(3) : 0.0;
  ego_vel.twist.twist.linear.z = n_dof_ > 4 ? ego_motion(4) : 0.0;
  ego_vel.twist.twist.angular.x = n_dof_ > 5 ? ego_motion(5) : 0.0;

  for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c)
    ego_vel.twist.covariance[6*r+c] = cov_mat(r, c);

  pcl::PointCloud<RadarPointType> inlier_points;
  pcl::PointCloud<RadarPointType> outlier_points;
  int filtered_size = scan_filtered->size();
  inlier_points.reserve(inlier_size);
  outlier_points.reserve(filtered_size - inlier_size);
  if (filtered_size > 0) {
    for (int i = 0, j = 0; i < filtered_size; ++i) {
      if (j < inlier_size && i == inlier_idx[j]) {
        inlier_points.push_back(scan_filtered->points[i]);
        j++;
      } else {
        outlier_points.push_back(scan_filtered->points[i]);
      }
    }
  }
  if(inlier_points.size() + outlier_points.size() != filtered_size){
    ROS_WARN("Something wrong during extracting the inliers");
  }

  pcl::PointCloud<RadarPointType> now_inliers;
  pcl::copyPointCloud(inlier_points, now_inliers);
  if (unstable_filter && initial) {
    unstable_point_detector(radar_vel, inlier_points);
  }

  sensor_msgs::PointCloud2 inlier_msg;
  pcl::toROSMsg(inlier_points, inlier_msg);
  inlier_msg.header.frame_id = "radar";
  inlier_msg.header.seq = msg_header.seq;
  inlier_msg.header.stamp = msg_header.stamp;
  sensor_msgs::PointCloud2 outlier_msg;
  pcl::toROSMsg(outlier_points, outlier_msg);
  outlier_msg.header.frame_id = "radar";
  outlier_msg.header.seq = msg_header.seq;
  outlier_msg.header.stamp = msg_header.stamp;

  sensor_msgs::PointCloud2 inlier_2d_msg;
  pcl::PointCloud<RadarPointType> inliers_2d;
  inliers_2d.reserve(inlier_points.size());
  for (auto p : inlier_points) {
    RadarPointType p_xy{};
    p_xy.data[3] = 1.0f;
    p_xy.azimuth = p.azimuth;
    p_xy.elevation = 0.0;
    p_xy.range = p.range;
    p_xy.rcs = p.rcs;
    p_xy.intensity = p.intensity;
    p_xy.x = p.x;
    p_xy.y = p.y;
    p_xy.z = 0.0;
    p_xy.velocity = p.velocity;
    p_xy.flag = p.flag;

    inliers_2d.push_back(p_xy);
  }
  pcl::toROSMsg(inliers_2d, inlier_2d_msg);
  inlier_2d_msg.header.frame_id = "radar";
  inlier_2d_msg.header.seq = msg_header.seq;
  inlier_2d_msg.header.stamp = msg_header.stamp;

  pub_inlier_2d_.publish(inlier_2d_msg);

  if (tictoc) {
    timer_radar_ego_motion.toc("radar ego motion");
    double ego_vel_time = timer_radar_ego_motion.toc_silent(); // 使用静默版本获取时间值
    ++total_frames_;
    total_ego_vel_time_ += ego_vel_time;

    double avg_ego_vel = total_ego_vel_time_ / total_frames_;
    std::cout << "平均自速度估计用时: " << avg_ego_vel << std::endl;
    if (log_file.is_open()) log_file << "平均自速度估计用时: " << avg_ego_vel << std::endl;
  }

  pub_ego_vel_.publish(ego_vel);
  pub_radar_vel_.publish(radar_vel);
  pub_inlier_.publish(inlier_msg);
  pub_outlier_.publish(outlier_msg);

  if (!initial) initial = true;
  last_radar_vel = radar_vel;
  last_ego_vel = ego_vel;
  last_inliers.clear();
  last_inliers.reserve(now_inliers.size());
  for (const auto &p : now_inliers) {
    RadarPointType point = p;
    point.flag = -1;
    last_inliers.push_back(point);
  }
}

void RadarEgoMotion::publishEmptyClouds() {
  pcl::PointCloud<RadarPointType> empty;
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(empty, msg);
  msg.header = msg_header;
  msg.header.frame_id = "radar";
  pub_inlier_.publish(msg); pub_outlier_.publish(msg); pub_inlier_2d_.publish(msg);
  initial = false;
  last_inliers.clear();
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "radar_ego_motion");
  try { RadarEgoMotion ego_motion; ros::spin(); }
  catch (const std::exception& error) { ROS_FATAL("Radar preprocessing failed: %s", error.what()); return 1; }

  return 0;
}
