#include <gtest/gtest.h>
#include "raplace.hpp"
#include "dyna_rc/raplace_input.h"
#include <pcl/conversions.h>

namespace {
cv::Mat pattern(int seed) {
  cv::Mat result(12, 18, CV_32F);
  cv::RNG(seed).fill(result, cv::RNG::NORMAL, 0., 1.);
  cv::Scalar mean, sd; cv::meanStdDev(result, mean, sd);
  return (result - mean[0]) / sd[0];
}
}

TEST(RaPlace, FixedMapIncludesLastFrameAndSingleFrame) {
  raplace::Params p; p.exclude_recent = 0;
  raplace::Matcher matcher(p);
  const auto first = pattern(10), last = pattern(20);
  int index; double distance;
  EXPECT_FALSE(matcher.query(last, &index, &distance));
  matcher.add(first);
  ASSERT_TRUE(matcher.query(first, &index, &distance));
  EXPECT_EQ(index, 0); EXPECT_NEAR(distance, 0., 1e-5);
  matcher.add(last);
  ASSERT_TRUE(matcher.query(last, &index, &distance));
  EXPECT_EQ(index, 1); EXPECT_NEAR(distance, 0., 1e-5);
}

TEST(RaPlace, CircularCorrelationAgreesWithDirectCalculation) {
  raplace::Params p; p.exclude_recent = 0;
  raplace::Matcher matcher(p);
  const auto query = pattern(42);
  std::vector<cv::Mat> db{pattern(1), pattern(2), pattern(3)};
  double expected = 1e10; int expected_index = -1;
  for (size_t i = 0; i < db.size(); ++i) {
    matcher.add(db[i]);
    double best = -1e10;
    for (int shift = 0; shift < query.cols; ++shift) {
      double corr = 0;
      for (int row = 0; row < query.rows; ++row) for (int col = 0; col < query.cols; ++col)
        corr += query.at<float>(row, col) * db[i].at<float>(row, (col + shift) % query.cols);
      best = std::max(best, corr);
    }
    const double distance = (cv::norm(query, cv::NORM_L2SQR) - best) / 1000.;
    if (distance < expected) { expected = distance; expected_index = i; }
  }
  int index; double distance;
  ASSERT_TRUE(matcher.query(query, &index, &distance));
  EXPECT_EQ(index, expected_index); EXPECT_NEAR(distance, expected, 1e-6);
}

TEST(RaPlace, ZeroSignalIsNotAPerfectMatchAndTiesAreStable) {
  raplace::Params p; p.exclude_recent = 0;
  raplace::Matcher matcher(p);
  const auto desc = pattern(7);
  matcher.add(desc); matcher.add(desc);
  int index; double distance;
  EXPECT_FALSE(matcher.query(cv::Mat::zeros(desc.size(), CV_32F), &index, &distance));
  for (int n = 0; n < 5; ++n) {
    ASSERT_TRUE(matcher.query(desc, &index, &distance)); EXPECT_EQ(index, 0);
  }
}

TEST(RaPlaceInput, MaximumIntensityAndPhysicalXYArePreserved) {
  raplace_input::Cloud cloud;
  auto add = [&](float x, float y, float intensity) {
    raplace_input::Point p; p.x=x; p.y=y; p.z=0; p.intensity=intensity; cloud.push_back(p);
  };
  add(2, 0, 10); add(2, 0, 30); add(-2, 0, 20); add(0, 2, 40); add(20, 0, 1000);
  const auto image = raplace_input::bev(cloud, 10, 1, 5, 0);
  EXPECT_EQ(image.at<float>(5,7), 30); EXPECT_EQ(image.at<float>(5,3), 20);
  EXPECT_EQ(image.at<float>(7,5), 40); EXPECT_EQ(cv::countNonZero(image), 3);
  raplace::Params params; params.work_size=10;
  raplace::Descriptor descriptor(params);
  const auto desc = descriptor.fromCart(image);
  EXPECT_TRUE(cv::checkRange(desc)); EXPECT_GT(cv::norm(desc), 0);
}

TEST(RaPlaceInput, FieldSelectionCannotSilentlyLoseRCS) {
  raplace_input::Cloud points;
  raplace_input::Point p; p.x=1; p.y=2; p.z=3; p.intensity=42; points.push_back(p);
  pcl::PCLPointCloud2 blob; pcl::toPCLPointCloud2(points, blob);
  for (auto& f : blob.fields) if (f.name == "intensity") f.name = "rcs";
  EXPECT_THROW(raplace_input::cloud(blob, "intensity", -2, 50), std::runtime_error);
  const auto result = raplace_input::cloud(blob, "rcs", -2, 50);
  ASSERT_EQ(result->size(), 1u); EXPECT_EQ(result->front().intensity, 42);
  EXPECT_TRUE(raplace_input::cloud(blob, "rcs", -2, 2)->empty());
}
