// seq_relocalization_enhanced.h
#ifndef SEQ_RELOCALIZATION_ENHANCED_H
#define SEQ_RELOCALIZATION_ENHANCED_H

#include "scancontext/Scancontext.h"
#include "dyna_rc/runtime_timing.h"
#include <deque>
#include <vector>
#include <random>
#include <Eigen/Dense>

class SeqRelocalizerEnhanced : public SCManager {
public:
    struct Particle {
        int start_idx;           // 起始帧索引
        double weight;           // 粒子权重
        int align_shift;         // 最佳对齐偏移量
        double seq_distance;     // 序列距离
        
        Particle() : start_idx(-1), weight(0.0), align_shift(0), seq_distance(1e9) {}
        Particle(int idx, double w) : start_idx(idx), weight(w), align_shift(0), seq_distance(1e9) {}
    };
    
    SeqRelocalizerEnhanced();
    ~SeqRelocalizerEnhanced();
    
    // 设置参数（借鉴 iLoc 的分层策略）
    void setParams(int window_size = 10, 
                   int num_exclude_recent = 30,
                   int coarse_window_size = 4,
                   int fine_window_size = 10,
                   double dist_thresh = 0.35);
    
    // 添加帧到窗口（支持多分辨率）
    void addFrameWithWindowMultiRes(pcl::PointCloud<SCPointType>& scan,
                                    pcl::PointCloud<SCPointType>& doppler);
    
    void addFrameWithWindowMultiRes(pcl::PointCloud<SCPointType>& scan);
    
    // 核心：粗到细的序列重定位（借鉴 iLoc Fast SeqMatch）
    bool relocalizeCoarseToFine(int& best_frame_id, 
                                float& yaw_diff_rad,
                                std::vector<int>& candidate_frames);
                                
    bool relocalizeCoarseToFineAligned(int& best_frame_id, 
                                float& yaw_diff_rad,
                                std::vector<int>& candidate_frames);
    
    // 粒子滤波序列匹配（可选，更鲁棒）
    bool relocalizeWithParticleFilter(int& best_frame_id, 
                                      float& yaw_diff_rad,
                                      int num_particles = 30);
    int getFineWindowSize() const { return window_size_fine_; }
    int getCurrentWindowSize() const { return window_multi_res_.sc_fine.size(); }

    bool buildSimilarityMatrix(
        Eigen::MatrixXf& similarity_matrix);

    bool relocalizeByConv(
        int& best_index,
        float& yaw_diff_rad,
        float& best_score);

    void initializeConvKernels(
        int window_size);

    float computeConvScore(
        const Eigen::MatrixXf& patch,
        const Eigen::MatrixXf& kernel);

    void setVotingParams(int top_k_single_frame, int top_m_coarse_candidates, int candidate_tolerance, bool skip);
    void setUseDynamicSC(bool enabled) { use_dynamic_sc_ = enabled; }

    std::vector<Eigen::MatrixXf> conv_kernels_;
    float getLastMatchScore() const { return last_match_score_; }
    const localization_runtime::SequenceTiming& getLastTiming() const { return last_timing_; }
    
private:
    localization_runtime::SequenceTiming last_timing_;
    bool databaseConsistent() const;
    bool use_dynamic_sc_ = true;

    // 多分辨率描述子存储
    struct MultiResDescriptor {
        std::deque<Eigen::MatrixXd> sc_coarse;      // 粗分辨率
        std::deque<Eigen::MatrixXd> mask_coarse;
        std::deque<Eigen::MatrixXd> sc_fine;        // 细分辨率
        std::deque<Eigen::MatrixXd> mask_fine;
    };
    
    MultiResDescriptor window_multi_res_;
    
    // 参数（借鉴 iLoc 的分层策略）
    int window_size_fine_;          // 细匹配窗口大小（默认10）
    int window_size_coarse_;        // 粗匹配窗口大小（默认4）
    int num_exclude_recent_;        // 排除最近帧数（默认30）
    int search_step_coarse_;        // 粗搜索步长（默认5）
    int search_step_fine_;          // 细搜索步长（默认1）
    double seq_dist_thresh_;        // 序列距离阈值（默认0.35）
    
    // 粒子滤波参数
    int num_particles_;
    double particle_variance_;      // 粒子初始化方差
    double weight_threshold_;       // 粒子权重阈值（低于此值剔除）
    
    // 辅助函数
    Eigen::MatrixXd downsampleScanContext(const Eigen::MatrixXd& sc, int factor);
    Eigen::MatrixXd downsampleMask(const Eigen::MatrixXd& mask, int factor);
    
    // 带偏移校正的序列距离计算（借鉴 iLoc 的线模型）
    double computeSequenceDistanceWithAlignment(
        const std::deque<Eigen::MatrixXd>& win1_sc,
        const std::deque<Eigen::MatrixXd>& win1_mask,
        const std::deque<Eigen::MatrixXd>& win2_sc,
        const std::deque<Eigen::MatrixXd>& win2_mask,
        int& best_global_shift);
    
    // 粒子权重更新（基于序列匹配度）
    void updateParticleWeights(std::vector<Particle>& particles,
                               const std::deque<Eigen::MatrixXd>& query_sc,
                               const std::deque<Eigen::MatrixXd>& query_mask);
    
    // 粒子重采样
    void resampleParticles(std::vector<Particle>& particles);
    
    // 粒子合并（借鉴 iLoc 的粒子合并策略）
    void mergeNearbyParticles(std::vector<Particle>& particles, int merge_threshold = 3);
    
    // 候选精细筛选（借鉴 iLoc 的候选精化）
    bool refineCandidate(int candidate_frame_id, 
                        int query_frame_id,
                        float& refined_yaw_diff,
                        double& confidence);
    
    // 辅助函数：角度转弧度
    inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

    void findSingleFrameTopK(const Eigen::MatrixXd& query_sc,
                             const Eigen::MatrixXd& query_mask,
                             int K,
                             std::vector<int>& candidate_indices,
                             int max_allowed_idx);

    int top_k_single_frame_ = 5;        // K
    int top_m_coarse_candidates_ = 5;   // M
    int candidate_tolerance_ = 2;       // 索引对齐容差（帧数）
    bool skip_fine_matching_ = false;   // 默认不跳过精细匹配
    float last_match_score_ = 0.0;      // 最近一次匹配的分数
};

#endif
