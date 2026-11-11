#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include "scancontext/Scancontext.h"

namespace fs = std::filesystem;
namespace {
constexpr double kPi = 3.14159265358979323846;
struct Frame {
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
  double yaw;
};

double contextDistance(SCManager& sc, size_t query, size_t reference, bool dynamic) {
  const double distance = dynamic
      ? sc.distanceBtnScanContextWithDynamic(sc.polarcontexts_[query], sc.polarcontexts_[reference],
                                            sc.dynamic_masks_[query], sc.dynamic_masks_[reference]).first
      : sc.distanceBtnScanContext(sc.polarcontexts_[query], sc.polarcontexts_[reference]).first;
  // Undefined distances may be returned as a large sentinel by the library.
  return std::isfinite(distance) && distance >= -1e-8 && distance <= 2.0 + 1e-8
      ? std::max(0.0, std::min(2.0, distance)) : std::numeric_limits<double>::quiet_NaN();
}

void generate(ros::NodeHandle& nh) {
  ros::NodeHandle private_nh("~");
  auto param = [&](const std::string& name, auto& value, const auto& fallback) {
    if (!private_nh.getParam(name, value)) nh.param(name, value, fallback);
  };  // Prefer node-private parameters; retain the supplied version's global parameters.
  std::string poses_file, pcd_dir, output_csv;
  bool dynamic;
  double radius, z_min, z_max, success_xy_m, success_yaw_deg;
  param("poses_file", poses_file, std::string("poses.txt"));
  param("pcd_dir", pcd_dir, std::string("."));
  param("output_csv", output_csv, std::string("non_neighbor.csv"));
  param("use_dynamic_sc", dynamic, false);
  param("sc_max_radius", radius, 80.0);
  param("z_min", z_min, -2.0);
  param("z_max", z_max, 50.0);
  param("success_xy_m", success_xy_m, 3.0);
  param("success_yaw_deg", success_yaw_deg, 15.0);
  if (!std::isfinite(radius) || radius <= 0 || !std::isfinite(z_min) || !std::isfinite(z_max) || z_min > z_max)
    throw std::runtime_error("Invalid sc_max_radius or z filter limits");
  if (!std::isfinite(success_xy_m) || success_xy_m < 0 || !std::isfinite(success_yaw_deg) || success_yaw_deg < 0 || success_yaw_deg > 180)
    throw std::runtime_error("Thresholds must be finite: success_xy_m >= 0 and success_yaw_deg in [0, 180]");
  if (output_csv.empty()) throw std::runtime_error("output_csv must not be empty");
  const auto destination = fs::absolute(output_csv).lexically_normal();
  if (destination == fs::absolute(poses_file).lexically_normal() ||
      (fs::exists(destination) && fs::equivalent(destination, poses_file)))
    throw std::runtime_error("output_csv must not overwrite the input pose file");

  std::ifstream input(poses_file);
  if (!input) throw std::runtime_error("Cannot open poses_file: " + poses_file);
  SCManager sc;
  sc.setMaximumRadius(radius);
  std::vector<Frame> frames;
  std::string line;
  size_t line_number = 0, fallback_count = 0;
  while (std::getline(input, line)) {
    if (!ros::ok()) throw std::runtime_error("Precomputation interrupted");
    ++line_number;
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    double stamp, x, y, z, qx, qy, qz, qw;
    std::istringstream row(line);
    if (!(row >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw))
      throw std::runtime_error("Invalid pose row " + std::to_string(line_number));
    Frame frame{Eigen::Vector3d(x, y, z), Eigen::Quaterniond(qw, qx, qy, qz), 0.0};
    const double norm = frame.orientation.norm();
    if (!std::isfinite(stamp) || !frame.position.allFinite() || !frame.orientation.coeffs().allFinite() || !std::isfinite(norm) || norm < 1e-12)
      throw std::runtime_error("Invalid pose/quaternion at row " + std::to_string(line_number));
    const auto q = frame.orientation.normalized();
    frame.yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));

    const std::string id = std::to_string(frames.size());
    const auto cloud_path = fs::path(pcd_dir) / (id + ".pcd");
    pcl::PointCloud<SCPointType> cloud, filtered, dynamic_cloud;
    if (pcl::io::loadPCDFile<SCPointType>(cloud_path.string(), cloud) < 0)
      throw std::runtime_error("Missing map PCD (indices must not shift): " + cloud_path.string());
    // Same z filter and descriptor construction as place_recognition.cpp; no ICP.
    filtered.reserve(cloud.size());
    for (const auto& point : cloud)
      if (point.z >= z_min && point.z <= z_max) filtered.push_back(point);
    const auto dynamic_path = fs::path(pcd_dir) / (id + "_dyna.pcd");
    if (dynamic && fs::exists(dynamic_path) && pcl::io::loadPCDFile<SCPointType>(dynamic_path.string(), dynamic_cloud) >= 0
        && !filtered.empty() && !dynamic_cloud.empty()) {
      sc.makeAndSaveScancontextAndKeysWithDynamic(filtered, dynamic_cloud);
    } else {
      sc.makeAndSaveScancontextAndKeys(filtered);
      if (dynamic) {
        ++fallback_count;
        if (sc.dynamic_masks_.size() < sc.polarcontexts_.size())
          sc.dynamic_masks_.push_back(Eigen::MatrixXd::Ones(sc.PC_NUM_RING, sc.PC_NUM_SECTOR));
      }
    }
    frames.push_back(frame);
  }
  const size_t n = frames.size();
  if (!n || sc.polarcontexts_.size() != n || (dynamic && sc.dynamic_masks_.size() != n))
    throw std::runtime_error("Empty map or inconsistent descriptor/mask counts");
  ROS_INFO("Loaded %zu keyframes; dynamic=%d, radius=%.3f, z=[%.3f, %.3f], static fallbacks=%zu",
           n, dynamic, radius, z_min, z_max, fallback_count);
  ROS_INFO("Non-neighbor definition: XY > %g m OR yaw > %g deg", success_xy_m, success_yaw_deg);

  fs::create_directories(destination.parent_path());
  const fs::path temporary = destination.string() + ".tmp";
  std::ofstream output(temporary);
  if (!output) throw std::runtime_error("Cannot write CSV: " + temporary.string());
  output << std::setprecision(17)
         << "# metric=non_neighbor_v1_min_cosine_" << (dynamic ? "dynamic" : "standard") << '\n'
         << "# sc_max_radius=" << radius << "\n# z_min=" << z_min << "\n# z_max=" << z_max
         << "\n# xy_threshold=" << success_xy_m << "\n# yaw_threshold_deg=" << success_yaw_deg << '\n'
         << "index,x,y,z,qx,qy,qz,qw,score,confuser_index,valid_comparisons\n";
  auto last_log = ros::WallTime::now();
  size_t missing = 0;
  for (size_t i = 0; i < n; ++i) {
    const auto& frame = frames[i];
    double best = std::numeric_limits<double>::infinity();
    int confuser = -1;
    size_t comparisons = 0;
    for (size_t j = 0; j < n; ++j) {
      if (!ros::ok()) throw std::runtime_error("Interrupted; partial CSV remains at " + temporary.string());
      if (i == j) continue;
      const double xy = (frame.position.head<2>() - frames[j].position.head<2>()).norm();
      const double delta = frame.yaw - frames[j].yaw;
      const double yaw_deg = std::abs(std::atan2(std::sin(delta), std::cos(delta))) * 180.0 / kPi;
      if (xy <= success_xy_m + 1e-9 && yaw_deg <= success_yaw_deg + 1e-9) continue;
      // Keep query/reference ordering: sector-alignment ties may make the approximation asymmetric.
      const double distance = contextDistance(sc, i, j, dynamic);
      if (!std::isfinite(distance)) continue;
      ++comparisons;
      if (distance < best) { best = distance; confuser = static_cast<int>(j); }
    }
    if (confuser < 0) { best = std::numeric_limits<double>::quiet_NaN(); ++missing; }
    output << i << ',' << frame.position.x() << ',' << frame.position.y() << ',' << frame.position.z() << ','
           << frame.orientation.x() << ',' << frame.orientation.y() << ',' << frame.orientation.z() << ',' << frame.orientation.w()
           << ',' << best << ',' << confuser << ',' << comparisons << '\n';
    if ((ros::WallTime::now() - last_log).toSec() >= 5.0 || i + 1 == n) {
      ROS_INFO("Non-neighbor precompute: %zu/%zu keyframes", i + 1, n);
      last_log = ros::WallTime::now();
    }
  }
  output.close();
  if (!output) throw std::runtime_error("Failed writing CSV: " + temporary.string());
  if (!ros::ok()) throw std::runtime_error("Interrupted; completed CSV was not installed");
  fs::rename(temporary, destination);  // Replace the previous table only after a complete write.
  ROS_INFO("Saved %zu rows to %s (%zu without a valid non-neighbor). Done; exiting.", n, destination.string().c_str(), missing);
}
}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "non_neighbor_precompute");
  ros::NodeHandle nh;
  try { generate(nh); }
  catch (const std::exception& error) { ROS_FATAL("Non-neighbor precompute failed: %s", error.what()); return 1; }
  return 0;  // Offline one-shot node: no subscriptions, publishers or spin loop.
}
