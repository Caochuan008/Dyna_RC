#include <gtest/gtest.h>
#include "dyna_rc/runtime_timing.h"

using namespace localization_runtime;

TEST(RuntimeTiming, RejectedQueriesRemainInDenominator) {
    Statistics stats;
    Sample sample;
    sample.stamp = 1;
    sample.total_ms = 10;
    sample.descriptor_ms = 2;
    stats.add(sample);  // Rejected: no ICP, but still a query.
    sample.stamp = 2;
    sample.total_ms = 30;
    sample.status = "published";
    sample.icp_ms = 12;
    sample.icp_calls = 1;
    stats.add(sample);
    EXPECT_EQ(stats.count, 2u);
    EXPECT_EQ(stats.published, 1u);
    EXPECT_DOUBLE_EQ(stats.mean(5), 6);
    EXPECT_DOUBLE_EQ(stats.mean(9), 20);
    EXPECT_DOUBLE_EQ(stats.stampRateHz(), 1);
}

TEST(RuntimeTiming, P95UsesNearestRankAndDoesNotReorderSamples) {
    Statistics stats;
    EXPECT_TRUE(std::isnan(stats.p95()));
    Sample sample;
    for (int i = 20; i > 0; --i) { sample.total_ms = i; stats.add(sample); }
    EXPECT_DOUBLE_EQ(stats.p95(), 19);
    EXPECT_DOUBLE_EQ(stats.totals.front(), 20);
    EXPECT_TRUE(std::isnan(stats.stampRateHz()));
}

TEST(RuntimeTiming, NestedCandidateCostsAreNotDoubleCounted) {
    Sample sample;
    sample.descriptor_ms = 1;
    sample.retrieval_ms = 2;
    sample.voting_ms = 3;
    sample.fine_ms = 4;
    sample.icp_ms = 12;
    sample.candidates_ms = 9;
    sample.candidate_icp_ms = 8;
    sample.total_ms = 25;
    EXPECT_DOUBLE_EQ(sample.values()[2], 7);
    EXPECT_DOUBLE_EQ(sample.values()[8], 3);  // 25 - 1 - 2 - 7 - 12
}

TEST(RuntimeTiming, StageTimerRecordsEarlyReturnAndStopsOnce) {
    double retrieval = 0, fine = 0;
    auto earlyReturn = [&] {
        StageTimer timer(retrieval);
        const auto start = Clock::now();
        while (elapsedMs(start) < 0.1) {}  // No sleep or simulated clock.
        timer.stop();
        const double stopped = retrieval;
        timer.stop();
        EXPECT_DOUBLE_EQ(retrieval, stopped);
        timer.next(fine);
        const auto fine_start = Clock::now();
        while (elapsedMs(fine_start) < 0.1) {}
        return;  // Destructor commits the active stage.
    };
    earlyReturn();
    EXPECT_GT(retrieval, 0);
    EXPECT_GT(fine, 0);
}
