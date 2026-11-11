#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace raplace_input {
using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;

template<class T> inline double scalar(const uint8_t* data) {
  T value;
  std::memcpy(&value, data, sizeof(value));
  return static_cast<double>(value);
}

inline int fieldSize(uint8_t type) {
  switch (type) {
    case pcl::PCLPointField::INT8: case pcl::PCLPointField::UINT8: return 1;
    case pcl::PCLPointField::INT16: case pcl::PCLPointField::UINT16: return 2;
    case pcl::PCLPointField::INT32: case pcl::PCLPointField::UINT32:
    case pcl::PCLPointField::FLOAT32: return 4;
    case pcl::PCLPointField::FLOAT64: return 8;
    default: throw std::runtime_error("Unsupported point field datatype");
  }
}

inline double value(const uint8_t* data, const pcl::PCLPointField& field) {
  data += field.offset;
  switch (field.datatype) {
    case pcl::PCLPointField::INT8: return scalar<int8_t>(data);
    case pcl::PCLPointField::UINT8: return scalar<uint8_t>(data);
    case pcl::PCLPointField::INT16: return scalar<int16_t>(data);
    case pcl::PCLPointField::UINT16: return scalar<uint16_t>(data);
    case pcl::PCLPointField::INT32: return scalar<int32_t>(data);
    case pcl::PCLPointField::UINT32: return scalar<uint32_t>(data);
    case pcl::PCLPointField::FLOAT32: return scalar<float>(data);
    case pcl::PCLPointField::FLOAT64: return scalar<double>(data);
    default: throw std::runtime_error("Unsupported point field datatype");
  }
}

// Explicit field selection avoids silently replacing absent intensity/RCS with zeros.
inline Cloud::Ptr cloud(const pcl::PCLPointCloud2& blob, const std::string& field_name,
                        double z_min, double z_max) {
  Cloud::Ptr out(new Cloud);
  if (!blob.width || !blob.height) return out;
  if (blob.is_bigendian) throw std::runtime_error("Big-endian clouds are not supported");
  auto field = [&](const std::string& name) {
    for (const auto& f : blob.fields) if (f.name == name) {
      if (!f.count || static_cast<size_t>(f.offset) + fieldSize(f.datatype) > blob.point_step)
        throw std::runtime_error("Invalid point field: " + name);
      return f;
    }
    throw std::runtime_error("Point cloud has no '" + name + "' field; check raplace_intensity_field");
  };
  const auto x = field("x"), y = field("y"), z = field("z"), intensity = field(field_name);
  if (blob.row_step < static_cast<size_t>(blob.width) * blob.point_step ||
      blob.data.size() < static_cast<size_t>(blob.height) * blob.row_step)
    throw std::runtime_error("Truncated point cloud data");
  out->reserve(static_cast<size_t>(blob.width) * blob.height);
  for (uint32_t row = 0; row < blob.height; ++row) for (uint32_t col = 0; col < blob.width; ++col) {
    const auto* data = blob.data.data() + static_cast<size_t>(row) * blob.row_step + col * blob.point_step;
    Point p;
    p.x = value(data, x); p.y = value(data, y); p.z = value(data, z);
    p.intensity = value(data, intensity);
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
        std::isfinite(p.intensity) && p.z >= z_min && p.z <= z_max) out->push_back(p);
  }
  out->is_dense = true;
  return out;
}

// Max intensity per XY cell; area averaging keeps thin returns when reducing size.
// Physical bearings are preserved, including unobserved space outside the radar FOV.
inline cv::Mat bev(const Cloud& cloud, int work_size, int oversample, double radius,
                   double intensity_offset) {
  const int side = work_size * oversample;
  cv::Mat image = cv::Mat::zeros(side, side, CV_32F);
  for (const auto& p : cloud) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.intensity) ||
        std::hypot(p.x, p.y) >= radius) continue;
    const int col = static_cast<int>(std::floor((p.x / radius + 1) * side / 2.0));
    const int row = static_cast<int>(std::floor((p.y / radius + 1) * side / 2.0));
    if (col < 0 || col >= side || row < 0 || row >= side) continue;
    float& cell = image.at<float>(row, col);
    cell = std::max(cell, static_cast<float>(p.intensity + intensity_offset));
  }
  cv::resize(image, image, cv::Size(work_size, work_size), 0, 0, cv::INTER_AREA);
  return image;
}
}  // namespace raplace_input
