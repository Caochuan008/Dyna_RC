#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct GroundTruth {
    ros::Time stamp;
    double tx, ty, tz;
    double qx, qy, qz, qw;
};

class GtPublisher {
public:
    GtPublisher(ros::NodeHandle& nh, const std::string& file_path)
        : nh_(nh), file_path_(file_path), idx_(0) {
        // 发布真值话题
        gt_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/gt_pose", 10);
        // 读取文件
        if (!loadData()) {
            ROS_ERROR("Failed to load ground truth data from %s", file_path_.c_str());
            ros::requestShutdown();
        }
    }

    bool loadData() {
        std::ifstream file(file_path_);
        if (!file.is_open()) {
            ROS_ERROR("Cannot open file: %s", file_path_.c_str());
            return false;
        }

        std::string line;
        int line_num = 0;
        double last_time = -1.0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue; // 跳过空行和注释
            std::istringstream iss(line);
            double t, tx, ty, tz, qx, qy, qz, qw;
            if (!(iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
                ROS_WARN("Invalid line %d: %s", line_num, line.c_str());
                continue;
            }
            // 检查时间是否严格递增（可选）
            if (t <= last_time) {
                ROS_WARN("Time not strictly increasing at line %d: %f -> %f", line_num, last_time, t);
            }
            last_time = t;

            GroundTruth gt;
            gt.stamp = ros::Time(t);  // 假设时间戳是秒（可含小数）
            gt.tx = tx; gt.ty = ty; gt.tz = tz;
            gt.qx = qx; gt.qy = qy; gt.qz = qz; gt.qw = qw;
            data_.push_back(gt);
            ++line_num;
        }
        file.close();
        ROS_INFO("Loaded %zu ground truth poses from %s", data_.size(), file_path_.c_str());
        return !data_.empty();
    }

    void run() {
        ros::Rate loop_rate(100);  // 100Hz 检查频率
        while (ros::ok()) {
            ros::Time now = ros::Time::now();
            // 等待仿真时间有效（use_sim_time = true 时，时间从0开始，收到/clock后才会 > 0）
            if (now.isZero()) {
                ROS_WARN_THROTTLE(1, "Waiting for valid simulation time (check use_sim_time and /clock)");
                ros::spinOnce();
                loop_rate.sleep();
                continue;
            }

            // 发布所有时间戳 <= 当前时间的真值
            while (idx_ < data_.size() && data_[idx_].stamp <= now) {
                publishPose(data_[idx_]);
                idx_++;
            }

            ros::spinOnce();
            loop_rate.sleep();
        }
    }

private:
    void publishPose(const GroundTruth& gt) {
        geometry_msgs::PoseStamped msg;
        msg.header.stamp = gt.stamp;
        msg.header.frame_id = "map";  // 可根据需要改为参数
        msg.pose.position.x = gt.tx;
        msg.pose.position.y = gt.ty;
        msg.pose.position.z = gt.tz;
        msg.pose.orientation.x = gt.qx;
        msg.pose.orientation.y = gt.qy;
        msg.pose.orientation.z = gt.qz;
        msg.pose.orientation.w = gt.qw;
        gt_pub_.publish(msg);
        ROS_DEBUG("Published pose at time %.6f", gt.stamp.toSec());
    }

    ros::NodeHandle nh_;
    ros::Publisher gt_pub_;
    std::string file_path_;
    std::vector<GroundTruth> data_;
    size_t idx_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gt_publisher");
    // 必须设置 use_sim_time 为 true，以便 ros::Time::now() 同步于 bag 的 /clock
    ros::NodeHandle nh("~");
    bool use_sim_time = true;
    nh.setParam("/use_sim_time", use_sim_time);

    std::string file_path;
    if (!nh.getParam("gt_file_path", file_path)) {
        ROS_ERROR("Parameter 'gt_file_path' not set. Usage: gt_file_path:=/path/to/gt.txt");
        return 1;
    }

    GtPublisher node(nh, file_path);
    node.run();
    return 0;
}