#include <gtest/gtest.h>
#include "dyna_rc/seq_relocalization_enhanced.h"

namespace {
pcl::PointCloud<SCPointType> scan() {
    pcl::PointCloud<SCPointType> cloud;
    for (int ring = 0; ring < 20; ++ring) {
        for (int sector = 0; sector < 20; ++sector) {
            const double angle = (-60.0 + (sector + 0.5) * 6.0) * M_PI / 180.0;
            const double radius = (ring + 0.5) * 4.0;
            SCPointType point;
            point.x = radius * std::cos(angle);
            point.y = radius * std::sin(angle);
            point.z = 0;
            point.intensity = 1 + (ring + sector * 2) % 9;
            cloud.push_back(point);
        }
    }
    return cloud;
}

pcl::PointCloud<SCPointType> indexedScan(int index) {
    auto cloud = scan();
    std::mt19937 random(index + 12345);
    std::uniform_real_distribution<float> intensity(1.0f, 100.0f);
    for (auto& point : cloud) point.intensity = intensity(random);
    return cloud;
}
}

TEST(SequenceStaticMode, StaticSaveAndCancelKeepMasksAligned) {
    SCManager database;
    auto cloud = scan();
    pcl::PointCloud<SCPointType> doppler;
    for (int i = 0; i < 7; ++i) doppler.push_back(cloud.front());
    database.makeAndSaveScancontextAndKeysWithDynamic(cloud, doppler);
    const Eigen::MatrixXd original = database.dynamic_masks_.front();
    EXPECT_LT(original.minCoeff(), 0.5);  // The original map mask has a blocked sector.
    database.makeAndSaveScancontextAndKeys(cloud);
    ASSERT_EQ(database.dynamic_masks_.size(), database.polarcontexts_.size());
    EXPECT_TRUE(database.dynamic_masks_.back().isOnes());
    database.CancelScancontextAndKeysWithDynamic();
    ASSERT_EQ(database.dynamic_masks_.size(), 1u);
    EXPECT_TRUE(database.dynamic_masks_.front().isApprox(original));
}

TEST(SequenceStaticMode, ObservationsNeverChangeMapDescriptorsOrMasks) {
    SeqRelocalizerEnhanced relocalizer;
    relocalizer.setParams(4, 0, 4, 4, 0.35);
    auto cloud = scan();
    pcl::PointCloud<SCPointType> doppler;
    relocalizer.makeAndSaveScancontextAndKeysWithDynamic(cloud, doppler);
    const Eigen::MatrixXd original = relocalizer.dynamic_masks_.front();
    for (int i = 0; i < 12; ++i) {
        relocalizer.addFrameWithWindowMultiRes(cloud);
        relocalizer.addFrameWithWindowMultiRes(cloud, doppler);
    }
    EXPECT_EQ(relocalizer.getCurrentWindowSize(), 4);
    ASSERT_EQ(relocalizer.polarcontexts_.size(), 1u);
    ASSERT_EQ(relocalizer.dynamic_masks_.size(), 1u);
    EXPECT_EQ(relocalizer.polarcontext_invkeys_mat_.size(), 1u);
    EXPECT_TRUE(relocalizer.dynamic_masks_.front().isApprox(original));
}

TEST(SequenceStaticMode, StaticAndDynamicModesMatchIdenticalSequences) {
    auto cloud = scan();
    pcl::PointCloud<SCPointType> doppler;
    for (bool dynamic : {false, true}) {
        SCOPED_TRACE(dynamic);
        SeqRelocalizerEnhanced relocalizer;
        relocalizer.setParams(4, 0, 4, 4, 0.35);
        relocalizer.setVotingParams(5, 5, 2, false);
        relocalizer.setUseDynamicSC(dynamic);
        for (int i = 0; i < 40; ++i) {
            if (dynamic) relocalizer.makeAndSaveScancontextAndKeysWithDynamic(cloud, doppler);
            else relocalizer.makeAndSaveScancontextAndKeys(cloud);
        }
        for (int i = 0; i < 4; ++i) {
            if (dynamic) relocalizer.addFrameWithWindowMultiRes(cloud, doppler);
            else relocalizer.addFrameWithWindowMultiRes(cloud);
        }
        int index = -1;
        float yaw = 0;
        std::vector<int> candidates;
        ASSERT_TRUE(relocalizer.relocalizeCoarseToFineAligned(index, yaw, candidates));
        EXPECT_GT(relocalizer.getLastTiming().retrieval_ms, 0);
        EXPECT_GT(relocalizer.getLastTiming().voting_ms, 0);
        EXPECT_GT(relocalizer.getLastTiming().fine_ms, 0);
        EXPECT_GE(index, 0);
        EXPECT_LT(index, 40);
        EXPECT_NEAR(relocalizer.getLastMatchScore(), 0., 1e-6);
        ASSERT_TRUE(relocalizer.relocalizeCoarseToFine(index, yaw, candidates));
        EXPECT_GE(index, 0);
        EXPECT_LT(index, 40);
        EXPECT_NEAR(relocalizer.getLastMatchScore(), 0., 1e-6);

        relocalizer.dynamic_masks_.pop_back();  // A broken database must fail, not dereference past its end.
        EXPECT_FALSE(relocalizer.relocalizeCoarseToFineAligned(index, yaw, candidates));
        EXPECT_DOUBLE_EQ(relocalizer.getLastTiming().voting_ms, 0);
        EXPECT_DOUBLE_EQ(relocalizer.getLastTiming().fine_ms, 0);
        EXPECT_FALSE(relocalizer.relocalizeCoarseToFine(index, yaw, candidates));
    }
}

TEST(SequenceRateAlignment, ReturnsNewestFrameForDifferentTraversalRates) {
    const int length = 30, endpoint = 110;
    for (bool dynamic : {false, true}) {
        for (double rate : {0.0, 0.25, 0.5, 0.6, 1.0, 1.5, 2.0, 3.25}) {
            SCOPED_TRACE(dynamic);
            SCOPED_TRACE(rate);
            SeqRelocalizerEnhanced relocalizer;
            relocalizer.setUseDynamicSC(dynamic);
            relocalizer.setParams(length, 0, 16, length, 0.1);
            relocalizer.setVotingParams(5, 5, 3, false);
            pcl::PointCloud<SCPointType> doppler;
            for (int i = 0; i < 130; ++i) {
                auto cloud = indexedScan(i);
                if (dynamic) relocalizer.makeAndSaveScancontextAndKeysWithDynamic(cloud, doppler);
                else relocalizer.makeAndSaveScancontextAndKeys(cloud);
            }
            for (int i = 0; i < length; ++i) {
                int map_index = endpoint - static_cast<int>(std::lround(rate * (length - 1 - i)));
                auto cloud = indexedScan(map_index);
                if (dynamic) relocalizer.addFrameWithWindowMultiRes(cloud, doppler);
                else relocalizer.addFrameWithWindowMultiRes(cloud);
            }
            int index = -1;
            float yaw = 0;
            std::vector<int> candidates;
            ASSERT_TRUE(relocalizer.relocalizeCoarseToFineAligned(index, yaw, candidates));
            EXPECT_EQ(index, endpoint);  // Never an arbitrary history frame or a fixed +L-1 offset.
            EXPECT_NEAR(yaw, 0.0, 1e-6);

            // Good history must not allow an unrelated newest observation through.
            auto unrelated = indexedScan(10000);
            if (dynamic) relocalizer.addFrameWithWindowMultiRes(unrelated, doppler);
            else relocalizer.addFrameWithWindowMultiRes(unrelated);
            EXPECT_FALSE(relocalizer.relocalizeCoarseToFineAligned(index, yaw, candidates));
            EXPECT_EQ(index, -1);
            EXPECT_FALSE(std::isfinite(relocalizer.getLastMatchScore()));
        }
    }
}

TEST(SequenceRateAlignment, RepeatedFrameAtMapStartAlsoWorksInCoarseOnlyMode) {
    for (bool skip_fine : {false, true}) {
        SeqRelocalizerEnhanced relocalizer;
        relocalizer.setUseDynamicSC(false);
        relocalizer.setParams(30, 0, 16, 30, 0.1);
        relocalizer.setVotingParams(5, 5, 3, skip_fine);
        for (int i = 0; i < 8; ++i) {
            auto cloud = indexedScan(i);
            relocalizer.makeAndSaveScancontextAndKeys(cloud);
        }
        auto cloud = indexedScan(0);
        for (int i = 0; i < 30; ++i) relocalizer.addFrameWithWindowMultiRes(cloud);
        int index = -1;
        float yaw = 1;
        std::vector<int> candidates;
        ASSERT_TRUE(relocalizer.relocalizeCoarseToFineAligned(index, yaw, candidates));
        EXPECT_EQ(index, 0);
        EXPECT_NEAR(yaw, 0.0, 1e-6);
        EXPECT_NEAR(relocalizer.getLastMatchScore(), 0.0, 1e-6);
    }
}
