// seq_relocalization_enhanced.cpp
#include "dyna_rc/seq_relocalization_enhanced.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <limits>
//iLoc reference
SeqRelocalizerEnhanced::SeqRelocalizerEnhanced() {
    // 默认参数
    setParams(10, 30, 4, 10, 0.35);
    num_particles_ = 30;
    particle_variance_ = 50.0;  // 初始搜索范围
    weight_threshold_ = 0.01;
}

SeqRelocalizerEnhanced::~SeqRelocalizerEnhanced() {
}

void SeqRelocalizerEnhanced::setParams(int window_size, 
                                       int num_exclude_recent,
                                       int coarse_window_size,
                                       int fine_window_size,
                                       double dist_thresh) {
    window_size_fine_ = fine_window_size;
    window_size_coarse_ = coarse_window_size;
    num_exclude_recent_ = num_exclude_recent;
    seq_dist_thresh_ = dist_thresh;
    search_step_coarse_ = std::max(1, coarse_window_size / 2);  // 粗搜索步长
    search_step_fine_ = 1;
}

void SeqRelocalizerEnhanced::setVotingParams(int top_k_single_frame,
                                             int top_m_coarse_candidates,
                                             int candidate_tolerance,
                                             bool skip) {
    top_k_single_frame_ = top_k_single_frame;
    top_m_coarse_candidates_ = top_m_coarse_candidates;
    candidate_tolerance_ = candidate_tolerance;
    skip_fine_matching_ = skip;
}

bool SeqRelocalizerEnhanced::databaseConsistent() const {
    const size_t n = polarcontexts_.size();
    if (dynamic_masks_.size() == n && polarcontext_invkeys_.size() == n &&
        polarcontext_vkeys_.size() == n && polarcontext_invkeys_mat_.size() == n)
        return true;
    std::cerr << "[Sequence] Descriptor/key/mask counts differ: descriptors=" << n
              << ", masks=" << dynamic_masks_.size() << "; skipping match." << std::endl;
    return false;
}

void SeqRelocalizerEnhanced::findSingleFrameTopK(const Eigen::MatrixXd& query_sc,
                                                 const Eigen::MatrixXd& query_mask,
                                                 int K,
                                                 std::vector<int>& candidate_indices,
                                                 int max_allowed_idx) {
    candidate_indices.clear();
    if (!polarcontext_tree_ || polarcontext_invkeys_to_search_.empty() || K <= 0 ||
        query_sc.rows() != PC_NUM_RING || query_sc.cols() != PC_NUM_SECTOR ||
        query_mask.rows() != query_sc.rows() || query_mask.cols() != query_sc.cols())
        return;

    // Query and map keys must use the same definition in static mode.
    Eigen::MatrixXd ringkey = use_dynamic_sc_
        ? makeRobustRingkeyFromScancontext(const_cast<Eigen::MatrixXd&>(query_sc),
                                          const_cast<Eigen::MatrixXd&>(query_mask))
        : makeRingkeyFromScancontext(const_cast<Eigen::MatrixXd&>(query_sc));
    if (!ringkey.allFinite()) return;
    std::vector<float> query_key = eig2stdvec(ringkey);

    // 2. KD‑Tree 搜索，获取较多候选（5 倍于正常数量，提高 Top‑K 精度）
    const int num_tree_candidates = std::min(
        static_cast<int>(polarcontext_invkeys_to_search_.size()),
        NUM_CANDIDATES_FROM_TREE * 5);

    std::vector<size_t> tree_candidates(num_tree_candidates);
    std::vector<float> out_dists_sqr(num_tree_candidates);
    nanoflann::KNNResultSet<float> knn_result(num_tree_candidates);
    knn_result.init(&tree_candidates[0], &out_dists_sqr[0]);
    polarcontext_tree_->index->findNeighbors(knn_result, &query_key[0], nanoflann::SearchParams(10));

    // 3. 对候选计算动态加权距离，限制索引范围
    std::vector<std::pair<double, int>> dists;
    for (size_t i = 0; i < knn_result.size(); ++i) {
        int idx = static_cast<int>(tree_candidates[i]);
        if (idx < 0 || idx > max_allowed_idx || static_cast<size_t>(idx) >= polarcontexts_.size() ||
            static_cast<size_t>(idx) >= dynamic_masks_.size()) continue;
        auto result = distanceBtnScanContextWithDynamic(
            const_cast<Eigen::MatrixXd&>(query_sc),
            polarcontexts_[idx],
            const_cast<Eigen::MatrixXd&>(query_mask),
            dynamic_masks_[idx]);
        if (std::isfinite(result.first)) dists.emplace_back(result.first, idx);
    }

    // 4. 排序取前 K 个
    std::sort(dists.begin(), dists.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    int K_actual = std::min(K, static_cast<int>(dists.size()));
    for (int i = 0; i < K_actual; ++i) {
        candidate_indices.push_back(dists[i].second);
    }
}

// 多分辨率下采样（借鉴 iLoc 的粗到细策略）
Eigen::MatrixXd SeqRelocalizerEnhanced::downsampleScanContext(const Eigen::MatrixXd& sc, int factor) {
    if (factor <= 1) return sc;
    
    int rows = sc.rows();
    int cols = sc.cols();
    int new_rows = rows / factor;
    int new_cols = cols / factor;
    
    Eigen::MatrixXd downsampled(new_rows, new_cols);
    
    for (int i = 0; i < new_rows; ++i) {
        for (int j = 0; j < new_cols; ++j) {
            double sum = 0.0;
            int count = 0;
            for (int di = 0; di < factor; ++di) {
                for (int dj = 0; dj < factor; ++dj) {
                    int orig_i = i * factor + di;
                    int orig_j = j * factor + dj;
                    if (orig_i < rows && orig_j < cols) {
                        sum += sc(orig_i, orig_j);
                        count++;
                    }
                }
            }
            downsampled(i, j) = (count > 0) ? (sum / count) : 0.0;
        }
    }
    return downsampled;
}

Eigen::MatrixXd SeqRelocalizerEnhanced::downsampleMask(const Eigen::MatrixXd& mask, int factor) {
    // 掩码使用最大值下采样（保留任何动态区域）
    if (factor <= 1) return mask;
    
    int rows = mask.rows();
    int cols = mask.cols();
    int new_rows = rows / factor;
    int new_cols = cols / factor;
    
    Eigen::MatrixXd downsampled = Eigen::MatrixXd::Zero(new_rows, new_cols);
    
    for (int i = 0; i < new_rows; ++i) {
        for (int j = 0; j < new_cols; ++j) {
            double max_val = 0.0;
            for (int di = 0; di < factor; ++di) {
                for (int dj = 0; dj < factor; ++dj) {
                    int orig_i = i * factor + di;
                    int orig_j = j * factor + dj;
                    if (orig_i < rows && orig_j < cols) {
                        max_val = std::max(max_val, mask(orig_i, orig_j));
                    }
                }
            }
            downsampled(i, j) = max_val;
        }
    }
    return downsampled;
}

void SeqRelocalizerEnhanced::addFrameWithWindowMultiRes(pcl::PointCloud<SCPointType>& scan,
                                                         pcl::PointCloud<SCPointType>& doppler) {
    // Online observations belong only to the query window; keep the map immutable.
    Eigen::MatrixXd sc_fine = this->makeScancontext(scan);
    Eigen::MatrixXd mask_fine = this->makeDynamicMaskFromDopplerCloud(doppler, scan);
    
    // 生成粗分辨率版本（借鉴 iLoc 的下采样）
    Eigen::MatrixXd sc_coarse = downsampleScanContext(sc_fine, 2);
    Eigen::MatrixXd mask_coarse = downsampleMask(mask_fine, 2);
    
    // 保存到多分辨率窗口
    window_multi_res_.sc_fine.push_back(sc_fine);
    window_multi_res_.mask_fine.push_back(mask_fine);
    window_multi_res_.sc_coarse.push_back(sc_coarse);
    window_multi_res_.mask_coarse.push_back(mask_coarse);
    
    // 保持窗口大小
    while ((int)window_multi_res_.sc_fine.size() > window_size_fine_) {
        window_multi_res_.sc_fine.pop_front();
        window_multi_res_.mask_fine.pop_front();
        window_multi_res_.sc_coarse.pop_front();
        window_multi_res_.mask_coarse.pop_front();
    }
}

void SeqRelocalizerEnhanced::addFrameWithWindowMultiRes(pcl::PointCloud<SCPointType>& scan) {
    // 生成当前帧描述子
    Eigen::MatrixXd sc_fine = this->makeScancontext(scan);
    
    // The distance function uses 1 = observable, 0 = masked out.
    Eigen::MatrixXd mask_fine = Eigen::MatrixXd::Ones(sc_fine.rows(), sc_fine.cols());
    
    // 生成粗分辨率版本
    Eigen::MatrixXd sc_coarse = downsampleScanContext(sc_fine, 2);
    Eigen::MatrixXd mask_coarse = downsampleMask(mask_fine, 2);
    
    // 保存到多分辨率窗口
    window_multi_res_.sc_fine.push_back(sc_fine);
    window_multi_res_.mask_fine.push_back(mask_fine);
    window_multi_res_.sc_coarse.push_back(sc_coarse);
    window_multi_res_.mask_coarse.push_back(mask_coarse);
    
    // 保持窗口大小
    while ((int)window_multi_res_.sc_fine.size() > window_size_fine_) {
        window_multi_res_.sc_fine.pop_front();
        window_multi_res_.mask_fine.pop_front();
        window_multi_res_.sc_coarse.pop_front();
        window_multi_res_.mask_coarse.pop_front();
    }
}

// 带偏移校正的序列距离计算（借鉴 iLoc 的线模型和粒子偏移）
double SeqRelocalizerEnhanced::computeSequenceDistanceWithAlignment(
        const std::deque<Eigen::MatrixXd>& win1_sc,
        const std::deque<Eigen::MatrixXd>& win1_mask,
        const std::deque<Eigen::MatrixXd>& win2_sc,
        const std::deque<Eigen::MatrixXd>& win2_mask,
        int& best_global_shift) {
    
    if (win1_sc.size() != win2_sc.size() || win1_sc.empty())
        return 1e9;
    
    int L = win1_sc.size();
    int num_shifts = PC_NUM_SECTOR;  // 偏航角偏移候选数
    
    // 存储每帧的最佳偏移和距离（借鉴 iLoc 的线模型）
    std::vector<int> best_shifts(L);
    std::vector<double> best_dists(L, 1e9);
    
    for (int i = 0; i < L; ++i) {
        Eigen::MatrixXd sc1 = win1_sc[i];
        Eigen::MatrixXd sc2 = win2_sc[i];
        Eigen::MatrixXd mask1 = win1_mask[i];
        Eigen::MatrixXd mask2 = win2_mask[i];
        
        // 搜索最佳列偏移
        double min_dist = 1e9;
        int best_shift = 0;
        
        for (int shift = 0; shift < num_shifts; ++shift) {
            // 使用父类的带掩码距离函数
            auto dist_pair = this->distanceBtnScanContextWithDynamic(sc1, sc2, mask1, mask2);
            double dist = dist_pair.first;
            int used_shift = dist_pair.second;
            
            if (dist < min_dist) {
                min_dist = dist;
                best_shift = used_shift;
            }
        }
        
        best_shifts[i] = best_shift;
        best_dists[i] = min_dist;
    }
    
    // 借鉴 iLoc：寻找最一致的偏移（线模型拟合）
    // 假设偏移量在序列中应该是连续的（斜率接近0）
    std::map<int, int> shift_votes;
    for (int i = 0; i < L; ++i) {
        shift_votes[best_shifts[i]]++;
    }
    
    // 找到投票最多的偏移作为全局偏移
    int max_votes = 0;
    best_global_shift = 0;
    for (const auto& vote : shift_votes) {
        if (vote.second > max_votes) {
            max_votes = vote.second;
            best_global_shift = vote.first;
        }
    }
    
    // 只计算与全局偏移一致的帧的距离
    double total_dist = 0.0;
    int valid_count = 0;
    for (int i = 0; i < L; ++i) {
        if (std::abs(best_shifts[i] - best_global_shift) <= 2) {  // 容忍±2的偏移误差
            total_dist += best_dists[i];
            valid_count++;
        }
    }
    
    return (valid_count > 0) ? (total_dist / valid_count) : 1e9;
}

// Variable-rate voting, anchored at the newest observation (seq_type == 2).
bool SeqRelocalizerEnhanced::relocalizeCoarseToFineAligned(int& best_frame_id,
                                                     float& yaw_diff_rad,
                                                     std::vector<int>& candidate_frames) {
    last_timing_ = {};
    localization_runtime::StageTimer stage(last_timing_.retrieval_ms);
    best_frame_id = -1;
    yaw_diff_rad = 0.0f;
    last_match_score_ = std::numeric_limits<float>::infinity();
    candidate_frames.clear();
    if (!databaseConsistent()) return false;
    const int L = window_size_fine_;
    const int count = static_cast<int>(polarcontexts_.size()) - std::max(0, num_exclude_recent_);
    if (L <= 0 || (int)window_multi_res_.sc_fine.size() < L || count <= 0 ||
        top_k_single_frame_ <= 0 || top_m_coarse_candidates_ <= 0)
        return false;

    polarcontext_invkeys_to_search_.assign(polarcontext_invkeys_mat_.begin(),
                                          polarcontext_invkeys_mat_.begin() + count);
    polarcontext_tree_.reset();
    polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING, polarcontext_invkeys_to_search_, 10);

    std::vector<std::vector<int>> topk_per_frame(L);
    for (int i = 0; i < L; ++i)
        findSingleFrameTopK(window_multi_res_.sc_fine[i],
                            window_multi_res_.mask_fine[i], top_k_single_frame_,
                            topk_per_frame[i], count - 1);

    stage.next(last_timing_.voting_ms);
    const int R = std::max(0, candidate_tolerance_);
    const int duration = std::max(1, L - 1);
    // span is the map-window travel in half-frame units; rate = span / (2 * duration).
    // Always map query[L-1] to end, even when map frames repeat or are skipped.
    auto mapIndex = [L, duration](int end, int span, int i) {
        return end - static_cast<int>(std::lround(0.5 * span * (L - 1 - i) / duration));
    };
    // Infer rate hypotheses from early-frame/current-frame matches; no fixed speed range.
    std::set<std::pair<int, int>> models;
    for (int end : topk_per_frame.back()) {
        models.emplace(end, 0);  // Stationary/repeated map frame.
        if (duration <= end) models.emplace(end, 2 * duration);
        for (int i = 0; i < (L + 1) / 2 && i < L - 1; ++i) {
            for (int c : topk_per_frame[i]) {
                if (c > end) continue;  // Forward traversal, as in the original matcher.
                int span = static_cast<int>(std::lround(2.0 * (end - c) * duration / (L - 1 - i)));
                if (span <= 2 * end) models.emplace(end, span);
            }
        }
    }

    // Cache pairwise distances shared by different endpoint/rate hypotheses.
    std::map<std::pair<int, int>, std::pair<double, int>> distances;
    auto distance = [&](int i, int k) -> const std::pair<double, int>& {
        auto key = std::make_pair(i, k);
        auto it = distances.find(key);
        if (it == distances.end()) {
            auto result = distanceBtnScanContextWithDynamic(
                window_multi_res_.sc_fine[i], polarcontexts_[k],
                window_multi_res_.mask_fine[i], dynamic_masks_[k]);
            it = distances.emplace(key, result).first;
        }
        return it->second;
    };
    struct CandidateVote {
        int end, span, votes, residual;
        double current_distance;
    };
    std::vector<CandidateVote> candidates_votes;
    for (const auto& model : models) {
        int votes = 0, residual = 0;
        for (int i = 0; i < L; ++i) {
            int expected = mapIndex(model.first, model.second, i), delta = R + 1;
            for (int c : topk_per_frame[i]) delta = std::min(delta, std::abs(c - expected));
            votes += delta <= R;
            residual += delta;
        }
        if (2 * votes >= L)
            candidates_votes.push_back({model.first, model.second, votes, residual,
                                       distance(L - 1, model.first).first});
    }
    if (candidates_votes.empty()) {
        std::cout << "[Voting] No supported endpoint/rate hypothesis." << std::endl;
        return false;
    }
    std::sort(candidates_votes.begin(), candidates_votes.end(),
              [](const CandidateVote& a, const CandidateVote& b) {
                  if (a.votes != b.votes) return a.votes > b.votes;
                  if (a.residual != b.residual) return a.residual < b.residual;
                  if (a.current_distance != b.current_distance) return a.current_distance < b.current_distance;
                  return std::make_pair(a.end, a.span) < std::make_pair(b.end, b.span);
              });
    std::vector<CandidateVote> top_candidates;
    for (const auto& c : candidates_votes) {
        if (std::any_of(top_candidates.begin(), top_candidates.end(), [&](const CandidateVote& kept) {
                return std::abs(c.end - kept.end) <= R && std::abs(c.span - kept.span) <= 2 * R;
            })) continue;
        top_candidates.push_back(c);
        candidate_frames.push_back(c.end);  // Candidates now identify the newest frame.
        if ((int)top_candidates.size() == top_m_coarse_candidates_) break;
    }

    // Coarse-only ablation still reports a real current-frame score and orientation.
    if (skip_fine_matching_) {
        for (const auto& c : top_candidates) {
            const auto& current = distance(L - 1, c.end);
            if (!std::isfinite(current.first) || current.first > seq_dist_thresh_) continue;
            best_frame_id = c.end;
            yaw_diff_rad = deg2rad(current.second * PC_UNIT_SECTORANGLE);
            last_match_score_ = current.first;
            return true;
        }
        return false;
    }

    stage.next(last_timing_.fine_ms);
    double best_distance = std::numeric_limits<double>::infinity(), best_current_distance = best_distance;
    double best_rate = 0.0;
    std::set<std::pair<int, int>> checked;
    for (const auto& c : top_candidates) {
        for (int end = std::max(0, c.end - R); end <= std::min(count - 1, c.end + R); ++end) {
            const auto& current = distance(L - 1, end);
            if (!std::isfinite(current.first) || current.first > seq_dist_thresh_) continue;
            for (int span = std::max(0, c.span - 2 * R); span <= std::min(2 * end, c.span + 2 * R); ++span) {
                if (!checked.emplace(end, span).second) continue;
                double sum = 0.0;
                int valid = 0;
                for (int i = 0; i < L; ++i) {
                    const auto& match = distance(i, mapIndex(end, span, i));
                    int shift = std::abs(match.second - current.second);
                    // Anchor yaw consistency to the current frame, including cyclic wraparound.
                    if (std::isfinite(match.first) && std::min(shift, PC_NUM_SECTOR - shift) <= 2) {
                        sum += match.first;
                        ++valid;
                    }
                }
                if (2 * valid < L) continue;
                double score = sum / valid;
                if (score > seq_dist_thresh_) continue;
                if (score < best_distance || (score == best_distance && current.first < best_current_distance)) {
                    best_distance = score;
                    best_current_distance = current.first;
                    best_rate = 0.5 * span / duration;
                    best_frame_id = end;
                    yaw_diff_rad = deg2rad(current.second * PC_UNIT_SECTORANGLE);
                }
            }
        }
    }
    if (best_frame_id < 0) {
        std::cout << "[Fine] No endpoint passed current-frame and sequence checks." << std::endl;
        return false;
    }
    last_match_score_ = best_distance;
    std::cout << "[Fine] Current frame -> map " << best_frame_id
              << ", map/query rate=" << best_rate << ", sequence distance=" << best_distance
              << ", current distance=" << best_current_distance << std::endl;
    return true;
}

bool SeqRelocalizerEnhanced::relocalizeCoarseToFine(int& best_frame_id,
                                                     float& yaw_diff_rad,
                                                     std::vector<int>& candidate_frames) {
    last_timing_ = {};
    localization_runtime::StageTimer stage(last_timing_.retrieval_ms);
    if (!databaseConsistent()) return false;
    // ===== 前置检查 =====
    const int L = window_size_fine_;               // 观测窗口长度 M
    if ((int)window_multi_res_.sc_fine.size() < L)
        return false;
    if ((int)this->polarcontexts_.size() < L + num_exclude_recent_)
        return false;

    candidate_frames.clear();

    // ===== 强制重建 KD‑Tree（与之前相同） =====
    int num_exclude = num_exclude_recent_;
    polarcontext_invkeys_to_search_.clear();
    auto begin = polarcontext_invkeys_mat_.begin();
    auto end = polarcontext_invkeys_mat_.end();
    if (polarcontext_invkeys_mat_.size() > num_exclude)
        end = polarcontext_invkeys_mat_.end() - num_exclude;
    polarcontext_invkeys_to_search_.assign(begin, end);
    polarcontext_tree_.reset();
    polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING, polarcontext_invkeys_to_search_, 10);

    // ===== 阶段1：单帧 Top‑K 检索 =====
    std::vector<std::vector<int>> topk_per_frame(L);
    int max_query_idx = (int)this->polarcontexts_.size() - num_exclude_recent_ - 1;

    for (int i = 0; i < L; ++i) {
        findSingleFrameTopK(window_multi_res_.sc_fine[i],
                            window_multi_res_.mask_fine[i],
                            top_k_single_frame_,
                            topk_per_frame[i],
                            max_query_idx);
    }

    stage.next(last_timing_.voting_ms);
    // ===== 阶段2：投票选出粗选候选 =====
    struct CandidateVote {
        int start_idx;
        int votes;
    };
    std::vector<CandidateVote> candidates_votes;
    int max_start = (int)this->polarcontexts_.size() - L;

    for (int s = 0; s <= max_start; s += search_step_coarse_) {
        if (s + L - 1 >= (int)this->polarcontexts_.size() - num_exclude_recent_)
            continue;

        int votes = 0;
        for (int i = 0; i < L; ++i) {
            int expected = s + i;
            for (int c : topk_per_frame[i]) {
                if (std::abs(c - expected) <= candidate_tolerance_) {
                    ++votes;
                    break;
                }
            }
        }
        if (votes > 0)
            candidates_votes.push_back({s, votes});
    }

    if (candidates_votes.empty()) {
        std::cout << "[Voting] No candidates after voting." << std::endl;
        return false;
    }

    std::sort(candidates_votes.begin(), candidates_votes.end(),
              [](const CandidateVote& a, const CandidateVote& b) { return a.votes > b.votes; });

    int M_vote = std::min(top_m_coarse_candidates_, (int)candidates_votes.size());
    std::vector<int> top_candidates;
    for (int i = 0; i < M_vote; ++i) {
        top_candidates.push_back(candidates_votes[i].start_idx);
        std::cout << "[Voting] Candidate " << i
                  << ": start=" << candidates_votes[i].start_idx
                  << ", votes=" << candidates_votes[i].votes << std::endl;
    }
    candidate_frames = top_candidates;
    if (skip_fine_matching_) {
        // 直接使用票数最高的候选作为结果（candidates_votes 已按票数降序排列）
        int best_start = top_candidates[0];                 // 票数最高的起始帧
        best_frame_id = best_start + L - 1;                 // 最后一帧的 id
        yaw_diff_rad = 0.0f;                                // 消融分析时 yaw 置零，或可简单设为 0
        std::cout << "[Voting] Skip fine matching. Best voted start = " << best_start
                  << " -> best_frame_id = " << best_frame_id << std::endl;
        return true;
    }

    // ===== 阶段3：改进的精细匹配（倒角式窗口距离） =====
    stage.next(last_timing_.fine_ms);
    std::cout << "[Fine] Starting fine matching (chamfer style) around "
              << top_candidates.size() << " candidates..." << std::endl;

    const int R = candidate_tolerance_;          // 邻域半径
    const int total_db = (int)this->polarcontexts_.size();
    const int valid_max = total_db - 1;         // 可访问的最大地图帧索引

    // 预存观测窗口的 SC 和 mask，避免重复取用
    std::vector<Eigen::MatrixXd> query_sc(L), query_mask(L);
    for (int i = 0; i < L; ++i) {
        query_sc[i] = window_multi_res_.sc_fine[i];
        query_mask[i] = window_multi_res_.mask_fine[i];
    }

    double best_aggregated_dist = 1e9;
    int best_map_frame = -1;            // 最终输出的地图帧ID
    int best_map_shift = 0;             // 对应的最佳列偏移

    for (int s : top_candidates) {
        // 邻域范围（确保不越界）
        int nb_start = std::max(0, s - R);
        int nb_end   = std::min(valid_max, s + R);

        double total_dist = 0.0;
        // 用于记录该候选区域内最佳的单个地图帧（可选，若最后决定用累加距离最小的区域作为定位，可以直接取该区域内最小单帧距离对应的帧）
        int local_best_frame = -1;
        double local_best_dist = 1e9;
        int local_best_shift = 0;

        for (int k = nb_start; k <= nb_end; ++k) {
            // 计算地图帧 k 与观测窗口所有帧的最小距离
            double min_dist_to_query = 1e9;
            int min_shift = 0;

            for (int i = 0; i < L; ++i) {
                auto result = distanceBtnScanContextWithDynamic(
                    query_sc[i], polarcontexts_[k],
                    query_mask[i], dynamic_masks_[k]);
                double dist = result.first;
                int shift = result.second;

                if (dist < min_dist_to_query) {
                    min_dist_to_query = dist;
                    min_shift = shift;
                }
            }

            // 累加该地图帧的贡献
            total_dist += min_dist_to_query;

            // 跟踪该区域内最优的单帧匹配（用于最终定位）
            if (min_dist_to_query < local_best_dist) {
                local_best_dist = min_dist_to_query;
                local_best_frame = k;
                local_best_shift = min_shift;
            }
        }
        total_dist = total_dist / float(nb_end- nb_start);

        std::cout << "   candidate s=" << s << "  aggregated dist=" << total_dist << std::endl;

        if (total_dist < best_aggregated_dist) {
            best_aggregated_dist = total_dist;
            best_map_frame = local_best_frame;
            best_map_shift = local_best_shift;
        }
    }

    // 检查最小距离是否在阈值内（可根据需要调整阈值含义，此处暂保留原有阈值）
    // 注意：现在 total_dist 是累加值，不能直接与 seq_dist_thresh_ 比较。
    // 你可以定义一个针对该累加距离的阈值，或者暂时移除此检查。
    if (best_map_frame < 0) {
        std::cout << "[Fine] No valid map frame found." << std::endl;
        return false;
    }

    best_frame_id = best_map_frame;
    yaw_diff_rad = deg2rad(best_map_shift * PC_UNIT_SECTORANGLE);
    last_match_score_ = best_aggregated_dist;

    std::cout << "[Fine] Relocalized to frame " << best_frame_id
              << " with aggregated dist " << best_aggregated_dist
              << ", yaw diff " << yaw_diff_rad * 180.0 / M_PI << " deg." << std::endl;

    return true;
}


//*******************************************************************************************************************************//

// // 核心：粗到细的序列重定位（完全借鉴 iLoc 的 Fast SeqMatch）
// bool SeqRelocalizerEnhanced::relocalizeCoarseToFine(int& best_frame_id, 
//                                                      float& yaw_diff_rad,
//                                                      std::vector<int>& candidate_frames) {
//     // 窗口未满
//     if ((int)window_multi_res_.sc_fine.size() < window_size_fine_)
//         return false;
    
//     // 数据库不足
//     if ((int)this->polarcontexts_.size() < window_size_fine_ + num_exclude_recent_)
//         return false;
    
//     candidate_frames.clear();
    
//     // ========== 阶段1: 粗匹配（使用下采样描述子，大步长）==========
//     std::cout << "[Coarse] Starting coarse matching..." << std::endl;
    
//     int max_start_coarse = (int)this->polarcontexts_.size() - window_size_coarse_;
//     double best_coarse_dist = 1e9;
//     std::vector<int> coarse_candidates;
//     std::vector<double> coarse_dists;
    
//     for (int start_idx = 0; start_idx <= max_start_coarse; start_idx += search_step_coarse_) {
//         // 排除最近帧
//         if (start_idx + window_size_coarse_ - 1 >= (int)this->polarcontexts_.size() - num_exclude_recent_)
//             continue;
        
//         // 构建粗分辨率历史窗口
//         std::deque<Eigen::MatrixXd> hist_sc_coarse, hist_mask_coarse;
//         for (int i = 0; i < window_size_coarse_; ++i) {
//             // 对历史帧也进行下采样
//             Eigen::MatrixXd sc_orig = this->polarcontexts_[start_idx + i];
//             Eigen::MatrixXd mask_orig = this->dynamic_masks_[start_idx + i];
//             hist_sc_coarse.push_back(downsampleScanContext(sc_orig, 2));
//             hist_mask_coarse.push_back(downsampleMask(mask_orig, 2));
//         }
        
//         // 获取当前粗窗口
//         std::deque<Eigen::MatrixXd> curr_sc_coarse, curr_mask_coarse;
//         for (int i = 0; i < window_size_coarse_; ++i) {
//             curr_sc_coarse.push_back(window_multi_res_.sc_coarse[i]);
//             curr_mask_coarse.push_back(window_multi_res_.mask_coarse[i]);
//         }
        
//         int dummy_shift;
//         double seq_dist = computeSequenceDistanceWithAlignment(curr_sc_coarse, curr_mask_coarse,
//                                                                 hist_sc_coarse, hist_mask_coarse,
//                                                                 dummy_shift);
        
//         if (seq_dist < best_coarse_dist) {
//             best_coarse_dist = seq_dist;
//         }
        
//         // 保留候选（阈值内）
//         if (seq_dist < seq_dist_thresh_ * 1.5) {  // 粗匹配更宽松
//             coarse_candidates.push_back(start_idx);
//             coarse_dists.push_back(seq_dist);
//         }
//     }
    
//     if (coarse_candidates.empty()) {
//         std::cout << "[Coarse] No candidates found" << std::endl;
//         return false;
//     }
    
//     // 按距离排序，保留前N个候选
//     std::vector<int> indices(coarse_candidates.size());
//     std::iota(indices.begin(), indices.end(), 0);
//     std::sort(indices.begin(), indices.end(), [&](int i, int j) {
//         return coarse_dists[i] < coarse_dists[j];
//     });
    
//     int num_coarse_keep = std::min(10, (int)coarse_candidates.size());
//     std::vector<int> top_coarse_candidates;
//     for (int i = 0; i < num_coarse_keep; ++i) {
//         top_coarse_candidates.push_back(coarse_candidates[indices[i]]);
//         std::cout << "[Coarse] Candidate " << i << ": start=" << coarse_candidates[indices[i]]
//                   << ", dist=" << coarse_dists[indices[i]] << std::endl;
//     }
    
//     // ========== 阶段2: 细匹配（使用原始分辨率，小步长）==========
//     std::cout << "[Fine] Starting fine matching around " << top_coarse_candidates.size() 
//               << " candidates..." << std::endl;
    
//     double best_fine_dist = 1e9;
//     int best_fine_start = -1;
//     int best_fine_shift = 0;
    
//     // 对每个粗候选，在其邻域内进行细搜索（借鉴 iLoc 的局部窗口）
//     for (int coarse_start : top_coarse_candidates) {
//         int search_start = std::max(0, coarse_start - window_size_coarse_);
//         int search_end = std::min(max_start_coarse, coarse_start + window_size_coarse_);
        
//         for (int start_idx = search_start; start_idx <= search_end; start_idx += search_step_fine_) {
//             // 排除最近帧
//             if (start_idx + window_size_fine_ - 1 >= (int)this->polarcontexts_.size() - num_exclude_recent_)
//                 continue;
            
//             // 构建精细历史窗口
//             std::deque<Eigen::MatrixXd> hist_sc_fine, hist_mask_fine;
//             for (int i = 0; i < window_size_fine_; ++i) {
//                 hist_sc_fine.push_back(this->polarcontexts_[start_idx + i]);
//                 hist_mask_fine.push_back(this->dynamic_masks_[start_idx + i]);
//             }
            
//             int global_shift;
//             double seq_dist = computeSequenceDistanceWithAlignment(window_multi_res_.sc_fine, 
//                                                                     window_multi_res_.mask_fine,
//                                                                     hist_sc_fine, hist_mask_fine,
//                                                                     global_shift);
            
//             if (seq_dist < best_fine_dist) {
//                 best_fine_dist = seq_dist;
//                 best_fine_start = start_idx;
//                 best_fine_shift = global_shift;
//             }
//         }
//     }
    
//     // 检查最佳匹配
//     if (best_fine_dist > seq_dist_thresh_) {
//         std::cout << "[Fine] Sequence distance too high: " << best_fine_dist 
//                   << " (thresh " << seq_dist_thresh_ << ")" << std::endl;
//         return false;
//     }
    
//     // ========== 阶段3: 候选精细筛选（借鉴 iLoc 的 Candidate Fine Selection）==========
//     best_frame_id = best_fine_start + window_size_fine_ - 1;
//     double refine_confidence = 0.0;
//     float refined_yaw = 0.0;
    
//     // 对最后几帧进行精细验证
//     int query_frame_id = window_multi_res_.sc_fine.size() - 1;
//     bool refine_success = refineCandidate(best_frame_id, query_frame_id, refined_yaw, refine_confidence);
    
//     if (refine_success && refine_confidence > 0.7) {
//         yaw_diff_rad = refined_yaw;
//         std::cout << "[Fine] Candidate refined with confidence: " << refine_confidence << std::endl;
//     } else {
//         yaw_diff_rad = deg2rad(best_fine_shift * PC_UNIT_SECTORANGLE);
//     }
    
//     std::cout << "[Fine] Relocalized to frame " << best_frame_id
//               << " with distance " << best_fine_dist
//               << ", yaw diff " << yaw_diff_rad * 180.0 / M_PI << " deg." << std::endl;
    
//     return true;
// }

// 粒子滤波序列匹配（借鉴 iLoc 的粒子滤波思想）
bool SeqRelocalizerEnhanced::relocalizeWithParticleFilter(int& best_frame_id, 
                                                           float& yaw_diff_rad,
                                                           int num_particles) {
    if ((int)window_multi_res_.sc_fine.size() < window_size_fine_)
        return false;
    
    if ((int)this->polarcontexts_.size() < window_size_fine_ + num_exclude_recent_)
        return false;
    
    // 初始化粒子
    std::vector<Particle> particles;
    int max_start = (int)this->polarcontexts_.size() - window_size_fine_;
    
    // 均匀采样粒子
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, max_start);
    
    for (int i = 0; i < num_particles; ++i) {
        int start_idx = dis(gen);
        // 排除最近帧
        if (start_idx + window_size_fine_ - 1 >= (int)this->polarcontexts_.size() - num_exclude_recent_)
            continue;
        particles.emplace_back(start_idx, 1.0 / num_particles);
    }
    
    // 迭代更新粒子权重（借鉴 iLoc 的多分辨率迭代）
    int max_iterations = 3;
    int current_resolution = 2;  // 从粗分辨率开始
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // 根据当前分辨率下采样描述子
        std::deque<Eigen::MatrixXd> query_sc, query_mask;
        int curr_window_size = (iter == 0) ? window_size_coarse_ : window_size_fine_;
        
        for (int i = 0; i < curr_window_size; ++i) {
            if (iter == 0 && i < (int)window_multi_res_.sc_coarse.size()) {
                query_sc.push_back(window_multi_res_.sc_coarse[i]);
                query_mask.push_back(window_multi_res_.mask_coarse[i]);
            } else if (i < (int)window_multi_res_.sc_fine.size()) {
                query_sc.push_back(window_multi_res_.sc_fine[i]);
                query_mask.push_back(window_multi_res_.mask_fine[i]);
            }
        }
        
        // 更新粒子权重
        updateParticleWeights(particles, query_sc, query_mask);
        
        // 剔除低权重粒子
        particles.erase(std::remove_if(particles.begin(), particles.end(),
            [this](const Particle& p) { return p.weight < weight_threshold_; }),
            particles.end());
        
        if (particles.empty()) break;
        
        // 粒子合并（借鉴 iLoc 的粒子合并策略）
        mergeNearbyParticles(particles, 5);
        
        // 重采样
        resampleParticles(particles);
        
        // 如果不是最后一次迭代，缩小搜索范围
        if (iter < max_iterations - 1) {
            // 找到最佳粒子的位置，重新初始化粒子围绕它
            auto best_particle = std::max_element(particles.begin(), particles.end(),
                [](const Particle& a, const Particle& b) { return a.weight < b.weight; });
            
            int center = best_particle->start_idx;
            int new_range = std::max(10, (max_start / (iter + 2)));
            std::uniform_int_distribution<> new_dis(std::max(0, center - new_range),
                                                    std::min(max_start, center + new_range));
            
            particles.clear();
            for (int i = 0; i < num_particles; ++i) {
                int start_idx = new_dis(gen);
                if (start_idx + window_size_fine_ - 1 >= (int)this->polarcontexts_.size() - num_exclude_recent_)
                    continue;
                particles.emplace_back(start_idx, 1.0 / num_particles);
            }
        }
    }
    
    // 找到最佳粒子
    if (particles.empty()) return false;
    
    auto best_particle = std::max_element(particles.begin(), particles.end(),
        [](const Particle& a, const Particle& b) { return a.weight < b.weight; });
    
    if (best_particle->seq_distance > seq_dist_thresh_)
        return false;
    
    best_frame_id = best_particle->start_idx + window_size_fine_ - 1;
    yaw_diff_rad = deg2rad(best_particle->align_shift * PC_UNIT_SECTORANGLE);
    
    std::cout << "[ParticleFilter] Relocalized to frame " << best_frame_id
              << " with weight " << best_particle->weight
              << ", yaw diff " << yaw_diff_rad * 180.0 / M_PI << " deg." << std::endl;
    
    return true;
}

void SeqRelocalizerEnhanced::updateParticleWeights(std::vector<Particle>& particles,
                                                    const std::deque<Eigen::MatrixXd>& query_sc,
                                                    const std::deque<Eigen::MatrixXd>& query_mask) {
    int L = query_sc.size();
    
    for (auto& particle : particles) {
        // 构建历史窗口
        std::deque<Eigen::MatrixXd> hist_sc, hist_mask;
        for (int i = 0; i < L; ++i) {
            if (particle.start_idx + i < (int)this->polarcontexts_.size()) {
                hist_sc.push_back(this->polarcontexts_[particle.start_idx + i]);
                hist_mask.push_back(this->dynamic_masks_[particle.start_idx + i]);
            }
        }
        
        if ((int)hist_sc.size() != L) {
            particle.weight = 0.0;
            continue;
        }
        
        // 计算带对齐的序列距离
        int best_shift;
        double seq_dist = computeSequenceDistanceWithAlignment(query_sc, query_mask,
                                                                hist_sc, hist_mask,
                                                                best_shift);
        
        particle.seq_distance = seq_dist;
        particle.align_shift = best_shift;
        
        // 权重与距离成反比
        particle.weight = std::exp(-seq_dist / seq_dist_thresh_);
    }
    
    // 归一化权重
    double sum_weights = 0.0;
    for (const auto& p : particles) sum_weights += p.weight;
    if (sum_weights > 0) {
        for (auto& p : particles) p.weight /= sum_weights;
    }
}

void SeqRelocalizerEnhanced::resampleParticles(std::vector<Particle>& particles) {
    if (particles.empty()) return;
    
    std::vector<Particle> new_particles;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // 轮盘赌选择
    for (size_t i = 0; i < particles.size(); ++i) {
        double r = dis(gen);
        double cumsum = 0.0;
        for (const auto& p : particles) {
            cumsum += p.weight;
            if (r < cumsum) {
                new_particles.push_back(p);
                break;
            }
        }
    }
    
    particles = new_particles;
}

void SeqRelocalizerEnhanced::mergeNearbyParticles(std::vector<Particle>& particles, int merge_threshold) {
    if (particles.empty()) return;
    
    // 按起始索引排序
    std::sort(particles.begin(), particles.end(),
              [](const Particle& a, const Particle& b) { return a.start_idx < b.start_idx; });
    
    std::vector<Particle> merged;
    for (size_t i = 0; i < particles.size(); ++i) {
        if (merged.empty() || std::abs(particles[i].start_idx - merged.back().start_idx) > merge_threshold) {
            merged.push_back(particles[i]);
        } else {
            // 合并：保留权重高的
            if (particles[i].weight > merged.back().weight) {
                merged.back() = particles[i];
            }
        }
    }
    
    particles = merged;
}

// 候选精细筛选（借鉴 iLoc 的 Candidate Fine Selection）
bool SeqRelocalizerEnhanced::refineCandidate(int candidate_frame_id, 
                                              int query_frame_id,
                                              float& refined_yaw_diff,
                                              double& confidence) {
    // 获取候选帧周围的局部窗口（±3帧）
    int window_radius = 3;
    int start_candidate = std::max(0, candidate_frame_id - window_radius);
    int end_candidate = std::min((int)this->polarcontexts_.size() - 1, candidate_frame_id + window_radius);
    
    // 获取查询帧的局部窗口
    int start_query = std::max(0, query_frame_id - window_radius);
    int end_query = std::min((int)window_multi_res_.sc_fine.size() - 1, query_frame_id + window_radius);
    
    double best_match_score = 0.0;
    int best_match_idx = candidate_frame_id;
    int best_shift = 0;
    
    // 在局部窗口内搜索最佳匹配
    for (int cand_idx = start_candidate; cand_idx <= end_candidate; ++cand_idx) {
        Eigen::MatrixXd cand_sc = this->polarcontexts_[cand_idx];
        Eigen::MatrixXd cand_mask = this->dynamic_masks_[cand_idx];
        
        for (int query_idx = start_query; query_idx <= end_query; ++query_idx) {
            Eigen::MatrixXd query_sc = window_multi_res_.sc_fine[query_idx];
            Eigen::MatrixXd query_mask = window_multi_res_.mask_fine[query_idx];
            
            // 计算距离和最佳偏移
            auto result = this->distanceBtnScanContextWithDynamic(query_sc, cand_sc,
                                                                   query_mask, cand_mask);
            double dist = result.first;
            int shift = result.second;
            
            // 将距离转换为相似度分数
            double score = std::exp(-dist / (seq_dist_thresh_ / 2.0));
            
            if (score > best_match_score) {
                best_match_score = score;
                best_match_idx = cand_idx;
                best_shift = shift;
            }
        }
    }
    
    confidence = best_match_score;
    
    if (confidence > 0.7) {
        refined_yaw_diff = deg2rad(best_shift * PC_UNIT_SECTORANGLE);
        return true;
    }
    
    return false;
}

