#include <gtest/gtest.h>
#include "dyna_rc/ntu_pointcloud_conversion.h"

namespace {
sensor_msgs::PointCloud frame() {
  sensor_msgs::PointCloud input;
  input.header.seq = 7; input.header.stamp = ros::Time(123, 456); input.header.frame_id = "oculii";
  input.points.resize(1); input.points[0].x=1; input.points[0].y=2; input.points[0].z=3;
  const char* names[] = {"Doppler", "Range", "Power", "Alpha", "Beta", "rcs"};
  const float values[] = {-2, 11, 32, 90, 30, -12};
  input.channels.resize(6);
  for (int i=0; i<6; ++i) { input.channels[i].name=names[i]; input.channels[i].values={values[i]}; }
  return input;
}
float at(const sensor_msgs::PointCloud2& output, size_t offset) {
  float value; std::memcpy(&value, output.data.data()+offset, sizeof(value)); return value;
}
}

TEST(NTUConversion, PreservesValidatedGeometryChannelsAndBinaryLayout) {
  const auto input = frame();
  sensor_msgs::PointCloud2 output; std::string error;
  ASSERT_TRUE(ntu_conversion::convert(input, output, error)) << error;
  EXPECT_EQ(output.header.stamp, input.header.stamp); EXPECT_EQ(output.header.seq, 7u);
  EXPECT_EQ(output.header.frame_id, "radar"); EXPECT_EQ(output.width, 1u); EXPECT_EQ(output.height, 1u);
  EXPECT_EQ(output.point_step, 96u); EXPECT_EQ(output.row_step, 96u); ASSERT_EQ(output.data.size(), 96u);
  ASSERT_EQ(output.fields.size(), 20u);
  for (size_t i=0; i<20; ++i) {
    EXPECT_EQ(output.fields[i].offset, 4*(i<3 ? i : i+1));
    EXPECT_EQ(output.fields[i].datatype, sensor_msgs::PointField::FLOAT32);
  }
  EXPECT_FLOAT_EQ(at(output,0), 1); EXPECT_FLOAT_EQ(at(output,4), 2); EXPECT_FLOAT_EQ(at(output,8), 3);
  EXPECT_FLOAT_EQ(at(output,16), 32); EXPECT_FLOAT_EQ(at(output,20), 11);
  EXPECT_NEAR(at(output,24), 1.57079632675, 1e-6); EXPECT_NEAR(at(output,28), -.52359877558, 1e-6);
  EXPECT_FLOAT_EQ(at(output,32), -2); EXPECT_FLOAT_EQ(at(output,40), -12);
  for (size_t offset : {12,36,44,48,52,56,60,64,68,72,76,80,84,88,92}) EXPECT_FLOAT_EQ(at(output,offset), 0);
}

TEST(NTUConversion, OptionalRCSDoesNotReplacePower) {
  auto input = frame(); input.channels.pop_back();
  sensor_msgs::PointCloud2 output; std::string error;
  ASSERT_TRUE(ntu_conversion::convert(input, output, error));
  EXPECT_FLOAT_EQ(at(output,16), 32); EXPECT_FLOAT_EQ(at(output,40), 0);
}

TEST(NTUConversion, RejectsShortOrMissingRequiredChannels) {
  auto input = frame(); input.channels[0].values.clear();
  sensor_msgs::PointCloud2 output; std::string error;
  EXPECT_FALSE(ntu_conversion::convert(input, output, error));
  EXPECT_NE(error.find("Doppler"), std::string::npos);
  input = frame(); input.channels.erase(input.channels.begin());
  EXPECT_FALSE(ntu_conversion::convert(input, output, error));
  EXPECT_NE(error.find("Doppler"), std::string::npos);
}

TEST(NTUConversion, EmptyAndNonfiniteScansHaveCorrectMetadata) {
  sensor_msgs::PointCloud input;
  sensor_msgs::PointCloud2 output; std::string error;
  ASSERT_TRUE(ntu_conversion::convert(input, output, error));
  EXPECT_TRUE(output.data.empty()); EXPECT_EQ(output.width, 0u);
  input = frame(); input.points[0].x = std::numeric_limits<float>::quiet_NaN();
  ASSERT_TRUE(ntu_conversion::convert(input, output, error));
  EXPECT_FALSE(output.is_dense); EXPECT_EQ(output.width, 1u);
}
