#pragma once

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ntu_conversion {
// Preserve the validated NTU mapping and the 20-field, 96-byte ZF-compatible layout.
inline bool convert(const sensor_msgs::PointCloud& input, sensor_msgs::PointCloud2& output,
                    std::string& error) {
  output = sensor_msgs::PointCloud2();
  error.clear();
  const size_t count = input.points.size();
  if (count > std::numeric_limits<uint32_t>::max() / 96) {
    error = "Point cloud is too large for PointCloud2.row_step"; return false;
  }
  const std::array<std::string, 6> channels = {{"Doppler", "Range", "Power", "Alpha", "Beta", "rcs"}};
  std::array<const std::vector<float>*, 6> values{};
  for (const auto& channel : input.channels)
    for (size_t j = 0; j < channels.size(); ++j)
      if (channel.name == channels[j]) values[j] = &channel.values;
  for (size_t j = 0; j < channels.size(); ++j) {
    // RCS is optional for NTU; the validated use_rcs=false path uses Power.
    if (count && !values[j] && j != 5) {
      error = "Missing NTU channel: " + channels[j]; return false;
    }
    if (values[j] && values[j]->size() < count) {
      error = "NTU channel shorter than points: " + channels[j]; return false;
    }
  }
  const char* names[] = {"x", "y", "z", "intensity", "range", "azimuth", "elevation",
                        "velocity", "snr", "rcs", "detectionconfidence", "mse_range", "mse_azimuth",
                        "mse_elevation", "mse_velocity", "std_dev_range", "std_dev_azimuthAngle",
                        "std_dev_elevationAngle", "std_dev_velocity", "std_dev_rcs"};
  output.header = input.header;
  output.header.frame_id = "radar";
  output.fields.resize(20);
  for (size_t j = 0; j < output.fields.size(); ++j) {
    auto& field = output.fields[j];
    field.name = names[j]; field.offset = 4 * (j < 3 ? j : j + 1);
    field.datatype = sensor_msgs::PointField::FLOAT32; field.count = 1;
  }
  output.height = 1; output.width = count;
  output.point_step = 96; output.row_step = output.point_step * output.width;
  output.is_bigendian = false; output.is_dense = true;
  output.data.resize(output.row_step, 0);
  constexpr double deg_to_rad = 3.1415926535 / 180.0;
  for (size_t i = 0; i < count; ++i) {
    const auto& p = input.points[i];
    float record[24]{};  // Zero all spare fields and padding, as in the supplied node.
    record[0] = p.x; record[1] = p.y; record[2] = p.z;
    record[4] = (*values[2])[i];                    // Power -> intensity
    record[5] = (*values[1])[i];                    // Range
    record[6] = (*values[3])[i] * deg_to_rad;        // Alpha -> azimuth (radians)
    record[7] = -(*values[4])[i] * deg_to_rad;       // -Beta -> elevation (radians)
    record[8] = (*values[0])[i];                    // Doppler sign unchanged
    record[10] = values[5] ? (*values[5])[i] : 0.f;  // Optional rcs
    for (float value : record) output.is_dense = output.is_dense && std::isfinite(value);
    std::memcpy(output.data.data() + i * output.point_step, record, sizeof(record));
  }
  return true;
}
}  // namespace ntu_conversion
