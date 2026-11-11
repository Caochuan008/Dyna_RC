#include <ros/ros.h>
#include "dyna_rc/ntu_pointcloud_conversion.h"
#include <algorithm>
#include <stdexcept>

class PointCloudConverter {
 public:
  PointCloudConverter() {
    // This private radar_topic is the raw PointCloud input; never read the global
    // /radar_topic, which existing localization launches use for /radar_v_points.
    pnh_.param<std::string>("radar_topic", input_topic_, "/radar_enhanced_pcl");
    pnh_.param<std::string>("output_topic", output_topic_, "/radar_v_points");
    if (!pnh_.getParam("debug_mode", debug_)) nh_.param("debug_mode", debug_, false);
    if (nh_.resolveName(input_topic_) == nh_.resolveName(output_topic_))
      throw std::runtime_error("PointCloud input and PointCloud2 output must use different topics");
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 10);
    sub_ = nh_.subscribe(input_topic_, 10, &PointCloudConverter::cloudCallback, this);
    ROS_INFO("NTU PointCloud converter: %s -> %s; Power -> intensity, Alpha/-Beta -> radians",
             input_topic_.c_str(), output_topic_.c_str());
  }

 private:
  void cloudCallback(const sensor_msgs::PointCloud::ConstPtr& input) {
    sensor_msgs::PointCloud2 output;
    std::string error;
    if (!ntu_conversion::convert(*input, output, error)) {
      ROS_ERROR_THROTTLE(5, "NTU conversion skipped: %s", error.c_str()); return;
    }
    pub_.publish(output);
    if (!debug_ || input->points.empty()) return;
    const std::vector<float>* velocities = nullptr;
    for (const auto& channel : input->channels)
      if (channel.name == "Doppler") velocities = &channel.values;
    if (!velocities) return;
    const auto end = velocities->begin() + input->points.size();
    const auto bounds = std::minmax_element(velocities->begin(), end);
    const double low = *bounds.first, high = *bounds.second;
    std::array<size_t, 4> bins{};
    for (auto it = velocities->begin(); it != end; ++it) {
      if (*it >= 0) ++bins[*it < high * .5 ? 0 : 1];
      else ++bins[*it < low * .5 ? 2 : 3];
    }
    const double percent = 100.0 / input->points.size();
    ROS_INFO("NTU Doppler: n=%zu, min=%.4f, max=%.4f m/s; [0,half max)=%.2f%%, [half max,max]=%.2f%%, "
             "[min,half min)=%.2f%%, [half min,0)=%.2f%%; negative=%.2f%%, nonnegative=%.2f%%",
             input->points.size(), low, high, bins[0]*percent, bins[1]*percent,
             bins[2]*percent, bins[3]*percent, (bins[2]+bins[3])*percent, (bins[0]+bins[1])*percent);
  }

  ros::NodeHandle nh_, pnh_{"~"};
  ros::Publisher pub_;
  ros::Subscriber sub_;
  std::string input_topic_, output_topic_;
  bool debug_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "pointcloud_converter");
  try { PointCloudConverter converter; ros::spin(); }
  catch (const std::exception& error) { ROS_FATAL("NTU converter failed: %s", error.what()); return 1; }
  return 0;
}
