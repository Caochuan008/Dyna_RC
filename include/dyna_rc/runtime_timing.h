#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace localization_runtime {
using Clock = std::chrono::steady_clock;  // Independent of /clock and bag playback rate.
inline double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Switching stages and early returns both commit elapsed time exactly once.
class StageTimer {
public:
    explicit StageTimer(double& output) : output_(&output), start_(Clock::now()) {}
    ~StageTimer() { stop(); }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;
    void stop() {
        if (output_) *output_ += elapsedMs(start_);
        output_ = nullptr;
    }
    void next(double& output) { stop(); output_ = &output; start_ = Clock::now(); }
private:
    double* output_;
    Clock::time_point start_;
};

struct SequenceTiming {
    double retrieval_ms = 0, voting_ms = 0, fine_ms = 0;
};

struct Sample {
    double stamp = 0;
    std::string kind = "query", status = "rejected";
    double descriptor_ms = 0, retrieval_ms = 0, voting_ms = 0, fine_ms = 0;
    double icp_ms = 0, candidates_ms = 0, candidate_icp_ms = 0, total_ms = 0;
    int icp_calls = 0, candidate_icp_calls = 0;
    std::array<double, 10> values() const {
        const double sequence = voting_ms + fine_ms;
        return {descriptor_ms, retrieval_ms, sequence, voting_ms, fine_ms, icp_ms,
                candidates_ms, candidate_icp_ms,
                std::max(0.0, total_ms - descriptor_ms - retrieval_ms - sequence - icp_ms), total_ms};
    }
};

inline constexpr std::array<const char*, 10> columns = {
    "descriptor", "retrieval", "sequence", "voting", "fine", "icp",
    "candidates", "candidate_icp", "other", "total"};

struct Statistics {
    size_t count = 0, published = 0, icp_calls = 0, candidate_icp_calls = 0;
    std::array<double, 10> sums{};
    std::vector<double> totals;
    double first_stamp = 0, last_stamp = 0;
    bool ordered_stamps = true;
    void add(const Sample& sample) {
        if (count == 0) first_stamp = sample.stamp;
        else if (sample.stamp <= last_stamp) ordered_stamps = false;
        last_stamp = sample.stamp;
        ++count;
        published += sample.status == "published";
        icp_calls += sample.icp_calls;
        candidate_icp_calls += sample.candidate_icp_calls;
        const auto values = sample.values();
        for (size_t i = 0; i < sums.size(); ++i) sums[i] += values[i];
        totals.push_back(sample.total_ms);
    }
    double mean(size_t i) const {
        return count ? sums[i] / count : std::numeric_limits<double>::quiet_NaN();
    }
    double p95() const {  // Nearest-rank percentile, including unsuccessful queries.
        if (totals.empty()) return std::numeric_limits<double>::quiet_NaN();
        auto sorted = totals;
        const size_t k = static_cast<size_t>(std::ceil(0.95 * sorted.size())) - 1;
        std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end());
        return sorted[k];
    }
    double stampRateHz() const {
        return count > 1 && ordered_stamps && last_stamp > first_stamp
            ? (count - 1) / (last_stamp - first_stamp) : std::numeric_limits<double>::quiet_NaN();
    }
};
}  // namespace localization_runtime
