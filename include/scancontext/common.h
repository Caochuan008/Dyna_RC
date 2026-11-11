#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include "dyna_rc/radar_point_type.h"

// typedef PointXYZRGBCov PointType;
typedef pcl::PointXYZI PointType;

using namespace Eigen;
using std::cout;
using std::endl;

// bool debug = false, tictoc = false, history_points_removal = false;

inline double rad2deg(double radians) {
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees) {
  return degrees * M_PI / 180.0;
}

inline float rad2deg(float radians) {
  return radians * 180.0 / M_PI;
}

inline float deg2rad(float degrees) {
  return degrees * M_PI / 180.0;
}

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

// sc param-independent helper functions
inline float xy2theta(const float &_x, const float &_y) {
  if ((_x >= 0) & (_y >= 0)) return (180 / M_PI) * atan(_y / _x)+60.0;

  if ((_x < 0) & (_y >= 0)) return 180 - ((180 / M_PI) * atan(_y / (-_x)))+60.0;

  if ((_x < 0) & (_y < 0)) return 180 + ((180 / M_PI) * atan(_y / _x))+60.0;

  if ((_x >= 0) & (_y < 0)) return 60 - ((180 / M_PI) * atan((-_y) / _x));
}  // xy2theta

inline std::vector<float> eig2stdvec(MatrixXd _eigmat) {
  std::vector<float> vec(_eigmat.data(), _eigmat.data() + _eigmat.size());
  return vec;
}  // eig2stdvec

MatrixXd circshift(MatrixXd &_mat, int _num_shift);
