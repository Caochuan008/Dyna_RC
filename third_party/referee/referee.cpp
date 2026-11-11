// Adapted from SPARO ReFeree (2024), BSD-3-Clause; see LICENSE and README.md.
#include "referee.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace referee {
void Params::validate() const {
  if (angle_bins < 1 || range_bins < 1 || angle_bins > 4096 || range_bins > 16384 ||
      static_cast<int64_t>(angle_bins) * range_bins > 16777216 ||
      range_block < 1 || angle_block < 1 || range_bins % range_block || angle_bins % angle_block)
    throw std::invalid_argument("ReFeree bins must be positive, divisible by their block sizes, and fit 16M cells");
}

PolarImage::PolarImage(const Params& p) : rows(p.angle_bins), cols(p.range_bins) {
  p.validate();
  cells.assign(static_cast<size_t>(rows) * cols, 0);
}

bool addPoint(PolarImage& image, double x, double y, double radius) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(radius) || radius <= 0) return false;
  const double r = std::hypot(x, y);
  if (!(r > 0 && r < radius)) return false;
  double angle = std::atan2(y, x);
  if (angle < 0) angle += 2 * pi;
  const int row = std::min(image.rows - 1, static_cast<int>(angle * image.rows / (2 * pi)));
  const int col = std::min(image.cols - 1, static_cast<int>(r * image.cols / radius));
  image.at(row, col) = 1;  // Repeated returns in one cell remain one feature.
  return true;
}

Descriptor describe(const PolarImage& image, const Params& p) {
  p.validate();
  if (image.rows != p.angle_bins || image.cols != p.range_bins ||
      image.cells.size() != static_cast<size_t>(image.rows) * image.cols)
    throw std::invalid_argument("ReFeree polar dimensions do not match parameters");
  Descriptor out;
  out.range.assign(p.range_bins / p.range_block, static_cast<double>(p.angle_bins) * p.range_block);
  out.angle.assign(p.angle_bins / p.angle_block, 0.0);
  for (int row = 0; row < image.rows; ++row) {
    int features = 0;
    for (int col = 0; col < image.cols; ++col) {
      if (!image.cells[static_cast<size_t>(row) * image.cols + col]) continue;
      --out.range[col / p.range_block];
      ++features;
    }
    out.occupied += features;
    // Match the released C++ A-ReFeree implementation: count cells from
    // the farthest return towards the range limit for each ray.
    bool farthest = false;
    for (int col = image.cols - 1; col > 0; --col) {
      if (!farthest && image.cells[static_cast<size_t>(row) * image.cols + col]) farthest = true;
      if (farthest) ++out.angle[row / p.angle_block];
    }
  }
  for (double& v : out.angle) v /= static_cast<double>(p.angle_block) * p.range_bins;
  return out;
}

bool Descriptor::valid() const {
  if (!occupied || range.empty() || angle.empty()) return false;
  for (const auto* values : {&range, &angle})
    for (double v : *values) if (!std::isfinite(v) || v < 0) return false;
  return true;
}

double distance(const Descriptor& q, const Descriptor& c) {
  if (!q.valid() || !c.valid() || q.range.size() != c.range.size())
    return std::numeric_limits<double>::infinity();
  double squared = 0;
  for (size_t i = 0; i < q.range.size(); ++i) {
    const double delta = q.range[i] - c.range[i];
    squared += delta * delta;
  }
  return std::sqrt(squared);
}

double queryToMapYaw(const Descriptor& q, const Descriptor& c) {
  if (!q.valid() || !c.valid() || q.angle.size() != c.angle.size()) return 0;
  double q2 = 0, c2 = 0;
  for (size_t i = 0; i < q.angle.size(); ++i) { q2 += q.angle[i]*q.angle[i]; c2 += c.angle[i]*c.angle[i]; }
  if (q2 <= 0 || c2 <= 0) return 0;  // No observable heading: identity ICP initialization.
  const int n = static_cast<int>(q.angle.size());
  int best_shift = 0;
  double best = -1;
  // Norms are unchanged by a circular shift: dot-product and cosine maxima agree.
  for (int shift = 0; shift < n; ++shift) {
    double dot = 0;
    for (int j = 0; j < n; ++j) dot += q.angle[j] * c.angle[(j + shift) % n];
    if (dot > best) { best = dot; best_shift = shift; }
  }
  const double yaw = best_shift * (2 * pi / n);
  return yaw > pi ? yaw - 2 * pi : yaw;
}
}  // namespace referee
