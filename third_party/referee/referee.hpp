// Adapted from SPARO ReFeree (2024), BSD-3-Clause; see LICENSE and README.md.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace referee {
constexpr double pi = 3.14159265358979323846;
struct Params {
  int angle_bins = 400, range_bins = 3360;
  int range_block = 80, angle_block = 8;
  void validate() const;
};
struct PolarImage {
  int rows, cols;
  std::vector<uint8_t> cells;
  explicit PolarImage(const Params& p);
  uint8_t& at(int row, int col) { return cells.at(static_cast<size_t>(row) * cols + col); }
};
struct Descriptor {
  std::vector<double> range, angle;
  size_t occupied = 0;
  bool valid() const;
};

// x forward, y left; CCW bearing in [0, 2*pi). Returns false outside the range.
bool addPoint(PolarImage& image, double x, double y, double radius);
Descriptor describe(const PolarImage& image, const Params& params);
// Native paper/Python metric: unnormalized Euclidean distance of free-cell counts.
double distance(const Descriptor& query, const Descriptor& candidate);
// Rotation to APPLY to query points to express them in the candidate sensor frame.
double queryToMapYaw(const Descriptor& query, const Descriptor& candidate);
}  // namespace referee
