#pragma once
#include "referee.hpp"
#include "dyna_rc/raplace_input.h"

// Reuse only the validated PointCloud2/PCD reader, not RaPlace's descriptor/BEV.
namespace referee_input {
using Cloud = raplace_input::Cloud;
using Point = raplace_input::Point;
inline referee::Descriptor describe(const Cloud& cloud, const referee::Params& params,
                                    double radius, double min_intensity) {
  referee::PolarImage polar(params);
  for (const auto& p : cloud)
    if (std::isfinite(p.z) && std::isfinite(p.intensity) && p.intensity >= min_intensity)
      referee::addPoint(polar, p.x, p.y, radius);
  return referee::describe(polar, params);
}
}  // namespace referee_input
