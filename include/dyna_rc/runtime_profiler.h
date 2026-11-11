#pragma once

#include "dyna_rc/runtime_timing.h"
#include <ros/ros.h>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <thread>

namespace localization_runtime {
class Profiler {
public:
    ~Profiler() { if (enabled_) report(); }

    void configure(const ros::NodeHandle& nh, const std::string& method, size_t map_size,
                   int window = 1, int stride = 1) {
        ros::NodeHandle private_nh("~");
        auto param = [&](const std::string& key, auto& value) {
            if (!private_nh.getParam(key, value)) nh.getParam(key, value);
        };
        param("runtime_timing_enabled", enabled_);
        if (!enabled_) return;
        param("runtime_report_every", report_every_);
        report_every_ = std::max(1, report_every_);
        const char* ros_home = std::getenv("ROS_HOME"), *home = std::getenv("HOME");
        std::string root = std::string(ros_home ? ros_home : (home ? home : "."));
        if (!ros_home) root += "/.ros";
        root += "/localization_runtime";
        param("runtime_output_dir", root);
        std::string name = method;
        param("runtime_run_name", name);
        if (name.empty()) name = method;
        for (char& c : name) if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '_';
        directory_ = root + "/" + name + "_" + std::to_string(ros::WallTime::now().toNSec());
        std::error_code error;
        std::filesystem::create_directories(directory_, error);
        if (!error) csv_.open(directory_ + "/frames.csv");
        if (error || !csv_.is_open()) {
            enabled_ = false;
            ROS_ERROR("Cannot write runtime measurements to %s: %s", directory_.c_str(), error.message().c_str());
            return;
        }
        csv_ << "frame,stamp,kind,status,icp_calls,candidate_icp_calls";
        for (const char* key : columns) csv_ << ',' << key << "_ms";
        csv_ << '\n' << std::fixed << std::setprecision(9);
        std::ofstream meta(directory_ + "/metadata.txt");
        meta << "method=" << method << "\nmap_size=" << map_size << "\nwindow_frames=" << window
             << "\ntrigger_stride_frames=" << stride << "\ncallback_threads=1\nhardware_threads="
             << std::thread::hardware_concurrency() << "\nclock=std::chrono::steady_clock\n"
             << "total_scope=validated query clouds to result publication or rejected attempt\n"
             << "icp_scope=all registrations including candidate outputs\n"
             << "excluded=offline map loading,input waiting,GT evaluation,debug cloud publishing,timing file IO\n";
        std::ifstream cpu("/proc/cpuinfo");
        for (std::string line; std::getline(cpu, line); )
            if (line.find("model name") == 0) { meta << "cpu_" << line << '\n'; break; }
        if (const char* omp = std::getenv("OMP_NUM_THREADS")) meta << "OMP_NUM_THREADS=" << omp << '\n';
        for (const char* key : {"poses_file", "pcd_dir", "use_dynamic_sc", "seq_type", "skip_fine_matching",
                                "sc_dist_thres", "seq_distance_threshold", "sc_max_radius", "z_min", "z_max",
                                "voxel_leaf_size", "max_corr_dist", "max_iter", "trans_eps", "fitness_eps",
                                "radar_topic", "radar_stat_topic", "radar_dyna_topic",
                                "raplace_input_mode", "raplace_intensity_field", "raplace_map_intensity_field",
                                "raplace_intensity_offset", "raplace_work_size", "raplace_bev_oversample",
                                "raplace_n_theta", "raplace_n_theta_out", "raplace_opencv_threads",
                                "raplace_sync_tolerance", "raplace_queue_size", "raplace_min_points"}) {
            XmlRpc::XmlRpcValue value;
            if (nh.getParam(key, value)) meta << key << '=' << value << '\n';
        }
        ROS_INFO("Runtime timing: %s; map=%zu, window=%d, stride=%d. Output: %s",
                 method.c_str(), map_size, window, stride, directory_.c_str());
    }

    void addICP(double ms) {
        if (active_) { active_->icp_ms += ms; ++active_->icp_calls; }
    }

    class Frame {
    public:
        Sample sample;
        Frame(Profiler& owner, double stamp) : owner_(owner), start_(Clock::now()) {
            sample.stamp = stamp;
            owner_.active_ = &sample;
        }
        ~Frame() { finish(); }
        Frame(const Frame&) = delete;
        Frame& operator=(const Frame&) = delete;
        void finish() {
            if (finished_) return;
            sample.total_ms = elapsedMs(start_);
            owner_.active_ = nullptr;
            finished_ = true;
            owner_.record(sample);  // CSV IO and summary calculation are outside the timer.
        }
    private:
        Profiler& owner_;
        Clock::time_point start_;
        bool finished_ = false;
    };

private:
    bool enabled_ = true;
    int report_every_ = 100;
    Sample* active_ = nullptr;  // These nodes use ros::spin(), so frames cannot overlap.
    std::string directory_;
    std::ofstream csv_;
    std::map<std::string, Statistics> stats_;

    void record(const Sample& sample) {
        if (!enabled_) return;
        stats_["all_frames"].add(sample);
        stats_[sample.kind].add(sample);
        if (sample.kind != "warmup") stats_["steady_frames"].add(sample);
        csv_ << stats_["all_frames"].count << ',' << sample.stamp << ',' << sample.kind << ','
             << sample.status << ',' << sample.icp_calls << ',' << sample.candidate_icp_calls;
        for (double value : sample.values()) csv_ << ',' << value;
        csv_ << '\n';
        csv_.flush();
        if (!csv_) { ROS_ERROR("Runtime frames.csv write failed (disk full or disconnected?)."); enabled_ = false; return; }
        if (stats_["all_frames"].count % report_every_ == 0) report();
    }

    void report() {
        if (stats_.empty()) return;
        std::ofstream summary(directory_ + "/summary.csv");
        summary << "scope,count,published,icp_calls,candidate_icp_calls,stamp_rate_hz";
        for (const char* key : columns) summary << ',' << key << "_mean_ms";
        summary << ",total_p95_ms\n" << std::fixed << std::setprecision(6);
        for (const auto& entry : stats_) {
            const auto& stat = entry.second;
            summary << entry.first << ',' << stat.count << ',' << stat.published << ','
                    << stat.icp_calls << ',' << stat.candidate_icp_calls << ',' << stat.stampRateHz();
            for (size_t i = 0; i < columns.size(); ++i) summary << ',' << stat.mean(i);
            summary << ',' << stat.p95() << '\n';
        }
        const auto it = stats_.find("query");
        if (it != stats_.end()) {
            const auto& s = it->second;
            ROS_INFO("[Runtime/query] n=%zu published=%zu | Descriptor %.3f Retrieval %.3f Sequence %.3f ICP %.3f | Total mean %.3f P95 %.3f ms",
                     s.count, s.published, s.mean(0), s.mean(1), s.mean(2), s.mean(5), s.mean(9), s.p95());
        }
        if (!summary) ROS_ERROR("Runtime summary.csv write failed.");
    }
};
}  // namespace localization_runtime
