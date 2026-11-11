#include "scancontext/Scancontext.h"

bool tictoc = false;

MatrixXd circshift(MatrixXd &_mat, int _num_shift) {
  // shift columns to right direction
  assert(_num_shift >= 0);

  if (_num_shift == 0) {
    MatrixXd shifted_mat(_mat);
    return shifted_mat;  // Early return
  }

  MatrixXd shifted_mat = MatrixXd::Zero(_mat.rows(), _mat.cols());
  for (int col_idx = 0; col_idx < _mat.cols(); col_idx++) {
    int new_location = (col_idx + _num_shift) % _mat.cols();
    shifted_mat.col(new_location) = _mat.col(col_idx);
  }

  return shifted_mat;
}  // circshift

double SCManager::distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2) {
  int num_eff_cols = 0;  // i.e., to exclude all-nonzero sector
  double sum_sector_similarity = 0;
  for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++) {
    VectorXd col_sc1 = _sc1.col(col_idx);
    VectorXd col_sc2 = _sc2.col(col_idx);

    if (col_sc1.norm() == 0 | col_sc2.norm() == 0)
      continue;  // don't count this sector pair.

    double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

    sum_sector_similarity = sum_sector_similarity + sector_similarity;
    num_eff_cols = num_eff_cols + 1;
  }

  double sc_sim = sum_sector_similarity / num_eff_cols;

  return 1.0 - sc_sim;
}  // distDirectSC

int SCManager::fastAlignUsingVkey(MatrixXd &_vkey1, MatrixXd &_vkey2) {
  int argmin_vkey_shift = 0;
  double min_veky_diff_norm = 10000000;
  for (int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++) {
    MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

    MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

    double cur_diff_norm = vkey_diff.norm();
    if (cur_diff_norm < min_veky_diff_norm) {
      argmin_vkey_shift = shift_idx;
      min_veky_diff_norm = cur_diff_norm;
    }
  }

  return argmin_vkey_shift;
}  // fastAlignUsingVkey

std::pair<double, int> SCManager::distanceBtnScanContext(MatrixXd &_sc1,
                                                         MatrixXd &_sc2) {
  // 1. fast align using variant key (not in original IROS18)
  MatrixXd vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
  MatrixXd vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
  int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);

  const int SEARCH_RADIUS =
      round(0.5 * SEARCH_RATIO * _sc1.cols());  // a half of search range
  std::vector<int> shift_idx_search_space{argmin_vkey_shift};
  for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++) {
    shift_idx_search_space.push_back((argmin_vkey_shift + ii + _sc1.cols()) %
                                     _sc1.cols());
    shift_idx_search_space.push_back((argmin_vkey_shift - ii + _sc1.cols()) %
                                     _sc1.cols());
  }
  std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

  // 2. fast columnwise diff
  int argmin_shift = 0;
  double min_sc_dist = 10000000;
  for (int num_shift : shift_idx_search_space) {
    MatrixXd sc2_shifted = circshift(_sc2, num_shift);
    double cur_sc_dist = distDirectSC(_sc1, sc2_shifted);
    if (cur_sc_dist < min_sc_dist) {
      argmin_shift = num_shift;
      min_sc_dist = cur_sc_dist;
    }
  }

  return make_pair(min_sc_dist, argmin_shift);
} // distanceBtnScanContext

MatrixXd SCManager::makeScancontext(pcl::PointCloud<SCPointType> &_scan_down) {
  TicTocV2 t_making_desc;

  int num_pts_scan_down = _scan_down.points.size();

  // main
  const int NO_POINT = -1000;
  MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

  SCPointType pt;
  float azim_angle, azim_range; // wihtin 2d plane
  int ring_idx, sctor_idx;
  for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++) {
    pt.x = _scan_down.points[pt_idx].x; 
    pt.y = _scan_down.points[pt_idx].y;
    pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).
    pt.intensity = _scan_down.points[pt_idx].intensity;

    // xyz to ring, sector
    azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
    azim_angle = xy2theta(pt.x, pt.y);

    // if range is out of roi, pass
    if (azim_range > PC_MAX_RADIUS) continue;

    ring_idx = std::max(std::min(PC_NUM_RING, int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
    sctor_idx = std::max(std::min(PC_NUM_SECTOR, int(ceil((azim_angle / 120.0) * PC_NUM_SECTOR))), 1);

    // taking maximum intensity
  //   if (desc(ring_idx - 1, sctor_idx - 1) < pt.intensity) // -1 means cpp starts from 0
  //     desc(ring_idx - 1, sctor_idx - 1) = pt.intensity; // update for taking maximum value at that bin
  // }
//     // taking maximum intensity
    if (desc(ring_idx - 1, sctor_idx - 1) < pt.z) // -1 means cpp starts from 0
      desc(ring_idx - 1, sctor_idx - 1) = pt.z; // update for taking maximum value at that bin
  }

  // reset no points to zero (for cosine dist later)
  for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
    for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
      if (desc(row_idx, col_idx) == NO_POINT)
        desc(row_idx, col_idx) = 0;

  if (tictoc) {
    t_making_desc.toc("PolarContext making");
  }

  return desc;
} // SCManager::makeScancontext

MatrixXd SCManager::makeScancontext_intensity(pcl::PointCloud<SCPointType> &_scan_down) {
  TicTocV2 t_making_desc;

  int num_pts_scan_down = _scan_down.points.size();

  // main
  const int NO_POINT = -1000;
  MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

  SCPointType pt;
  float azim_angle, azim_range; // wihtin 2d plane
  int ring_idx, sctor_idx;
  for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++) {
    pt.x = _scan_down.points[pt_idx].x; 
    pt.y = _scan_down.points[pt_idx].y;
    pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).
    pt.intensity = _scan_down.points[pt_idx].intensity;

    // xyz to ring, sector
    azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
    azim_angle = xy2theta(pt.x, pt.y);

    // if range is out of roi, pass
    if (azim_range > PC_MAX_RADIUS) continue;

    ring_idx = std::max(std::min(PC_NUM_RING, int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
    sctor_idx = std::max(std::min(PC_NUM_SECTOR, int(ceil((azim_angle / 120.0) * PC_NUM_SECTOR))), 1);

    // taking maximum intensity
    if (desc(ring_idx - 1, sctor_idx - 1) < pt.intensity) // -1 means cpp starts from 0
      desc(ring_idx - 1, sctor_idx - 1) = pt.intensity; // update for taking maximum value at that bin
  }
//     // taking maximum intensity
  //   if (desc(ring_idx - 1, sctor_idx - 1) < pt.z) // -1 means cpp starts from 0
  //     desc(ring_idx - 1, sctor_idx - 1) = pt.z; // update for taking maximum value at that bin
  // }

  // reset no points to zero (for cosine dist later)
  for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
    for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
      if (desc(row_idx, col_idx) == NO_POINT)
        desc(row_idx, col_idx) = 0;

  if (tictoc) {
    t_making_desc.toc("PolarContext making");
  }

  return desc;
}

MatrixXd SCManager::makeRingkeyFromScancontext(Eigen::MatrixXd &_desc) {
  /* 
    * summary: rowwise mean vector
  */
  Eigen::MatrixXd invariant_key(_desc.rows(), 1);
  for (int row_idx = 0; row_idx < _desc.rows(); row_idx++) {
    Eigen::MatrixXd curr_row = _desc.row(row_idx);
    invariant_key(row_idx, 0) = curr_row.mean();
  }

  return invariant_key;
} // SCManager::makeRingkeyFromScancontext

MatrixXd SCManager::makeSectorkeyFromScancontext(Eigen::MatrixXd &_desc) {
  /* 
   * summary: columnwise mean vector
   */
  Eigen::MatrixXd variant_key(1, _desc.cols());
  for (int col_idx = 0; col_idx < _desc.cols(); col_idx++) {
      Eigen::MatrixXd curr_col = _desc.col(col_idx);
      variant_key(0, col_idx) = curr_col.mean();
  }

  return variant_key;
} // SCManager::makeSectorkeyFromScancontext

void SCManager::makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> &_scan_down) {
  Eigen::MatrixXd sc;
  if (SC_CHANNEL == 1){
    sc = makeScancontext(_scan_down);  // v1
  } 
  if (SC_CHANNEL == 2){
    sc = makeScancontext_intensity(_scan_down);  // v2
  } 
  Eigen::MatrixXd ringkey = makeRingkeyFromScancontext(sc);
  Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext(sc);
  std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);

  polarcontexts_.push_back(sc);
  polarcontext_invkeys_.push_back(ringkey);
  polarcontext_vkeys_.push_back(sectorkey);
  polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);
  // Keep masks aligned with descriptors, including static-only/fallback frames.
  dynamic_masks_.push_back(Eigen::MatrixXd::Ones(sc.rows(), sc.cols()));
}  // SCManager::makeAndSaveScancontextAndKeys

void SCManager::setSCdistThres(double _new_thres) {
  SC_DIST_THRES = _new_thres;
} // SCManager::setThres

void SCManager::setMaximumRadius(double _max_r) {
  PC_MAX_RADIUS = _max_r;
} // SCManager::setMaximumRadius

void SCManager::setScChannel(int _sc_channel) {
  SC_CHANNEL = _sc_channel;
} // SCManager::setThres

std::pair<int, float> SCManager::detectLoopClosureID(void) {
  int loop_id{-1}; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")
  top_candidates_.clear();

  auto curr_key = polarcontext_invkeys_mat_.back(); // current observation (query)
  auto curr_desc = polarcontexts_.back(); // current observation (query)

  /* 
   * step 1: candidates from ringkey tree_
   */
  if ((int)polarcontext_invkeys_mat_.size() < NUM_EXCLUDE_RECENT + 1) {
    std::pair<int, float> result{loop_id, 0.0};
    return result; // Early return
  }

  // tree_ reconstruction (not mandatory to make everytime)
  if (tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) { // to save computation cost
    TicTocV2 t_tree_construction;

    polarcontext_invkeys_to_search_.clear();
    polarcontext_invkeys_to_search_.assign(polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end() - NUM_EXCLUDE_RECENT);

    polarcontext_tree_.reset(); 
    polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */);
    // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
    t_tree_construction.toc("Tree construction");
  }
  tree_making_period_conter = tree_making_period_conter + 1;
        
  double min_dist = 10000000; // init with somthing large
  int nn_align = 0;
  int nn_idx = 0;

  // knn search
  std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE); 
  std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);

  TicTocV2 t_tree_search;
  nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_FROM_TREE);
  knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
  polarcontext_tree_->index->findNeighbors(knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10)); 
  if (tictoc) {
    t_tree_search.toc("Tree search");
  }

  for (int i = 0; i < NUM_CANDIDATES_FROM_TREE; ++i) {
        top_candidates_.emplace_back(
            static_cast<int>(candidate_indexes[i]),
            out_dists_sqr[i]
        );
    }

  /*
   *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
   */
  TicTocV2 t_calc_dist;
  for (int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++) {
    MatrixXd polarcontext_candidate = polarcontexts_[candidate_indexes[candidate_iter_idx]];
    std::pair<double, int> sc_dist_result = distanceBtnScanContext(curr_desc, polarcontext_candidate);

    double candidate_dist = sc_dist_result.first;
    int candidate_align = sc_dist_result.second;

    if (candidate_dist < min_dist) {
      min_dist = candidate_dist;
      nn_align = candidate_align;

      nn_idx = candidate_indexes[candidate_iter_idx];
    }
  }
  if (tictoc) {
    t_calc_dist.toc("Distance calc");
  }

  /*
   * loop threshold check
   */
  if (min_dist < SC_DIST_THRES) {
    loop_id = nn_idx;
    
    // std::cout.precision(3);
    std::cout << "[Loop found] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << std::endl;
    // std::cout << "[Loop found] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg." << std::endl;
  } else {
    std::cout.precision(3);
    std::cout << "[Not loop] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << std::endl;
    // std::cout << "[Not loop] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg." << std::endl;
  }

  // To do: return also nn_align (i.e., yaw diff)
  float yaw_diff_rad = deg2rad(nn_align * PC_UNIT_SECTORANGLE);
  std::pair<int, float> result{loop_id, min_dist};

  return result;
} // SCManager::detectLoopClosureID

//********--------new--------********
/**
 * @brief 从动态点云生成动态标签矩阵
 * @param _doppler_cloud 包含多普勒速度的点云（同一帧）
 * @param _scan_down 原始点云（用于生成Scan Context的同一帧点云）
 * @return 动态标签矩阵，1表示静态（可观测），0表示动态（不可观测）
 */
Eigen::MatrixXd SCManager::makeDynamicMaskFromDopplerCloud(
    pcl::PointCloud<SCPointType> &_doppler_cloud,
    pcl::PointCloud<SCPointType> &_scan_down) {
    
    // 初始化动态标签矩阵，默认为1（全部可观测）
    Eigen::MatrixXd dynamic_mask = Eigen::MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);
    
    // 统计每个扇区-环的动态点数量及总点数
    Eigen::MatrixXd dynamic_count = Eigen::MatrixXd::Zero(PC_NUM_RING, PC_NUM_SECTOR);
    Eigen::MatrixXd total_count = Eigen::MatrixXd::Zero(PC_NUM_RING, PC_NUM_SECTOR);
    
    SCPointType pt;
    float azim_angle, azim_range;
    int ring_idx, sector_idx;
    
    // 统计原始点云中每个格子的总点数
    for (size_t pt_idx = 0; pt_idx < _scan_down.points.size(); pt_idx++) {
        pt.x = _scan_down.points[pt_idx].x; 
        pt.y = _scan_down.points[pt_idx].y;
        pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT;
        
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        azim_angle = xy2theta(pt.x, pt.y);
        if (azim_range > PC_MAX_RADIUS) continue;
        
        ring_idx = std::max(std::min(PC_NUM_RING, 
            int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
        sector_idx = std::max(std::min(PC_NUM_SECTOR, 
            int(ceil((azim_angle / 120.0) * PC_NUM_SECTOR))), 1);
        
        total_count(ring_idx-1, sector_idx-1) += 1;
    }
    
    // 统计动态点云中每个格子的动态点数量
    for (size_t pt_idx = 0; pt_idx < _doppler_cloud.points.size(); pt_idx++) {
        pt.x = _doppler_cloud.points[pt_idx].x; 
        pt.y = _doppler_cloud.points[pt_idx].y;
        pt.z = _doppler_cloud.points[pt_idx].z + LIDAR_HEIGHT;
        
        float doppler_velocity = _doppler_cloud.points[pt_idx].intensity;
        if (doppler_velocity < 0) continue; // 只考虑动态点
        
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        azim_angle = xy2theta(pt.x, pt.y);
        if (azim_range > PC_MAX_RADIUS) continue;
        
        ring_idx = std::max(std::min(PC_NUM_RING, 
            int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
        sector_idx = std::max(std::min(PC_NUM_SECTOR, 
            int(ceil((azim_angle / 120.0) * PC_NUM_SECTOR))), 1);
        
        dynamic_count(ring_idx-1, sector_idx-1) += 1;
    }
    
    // 对每个扇区，确定动态物体存在的最近环，并将其后所有环标记为遮挡
    const double DYNAMIC_RATIO_THRESH = 0.8;

    for (int sector = 0; sector < PC_NUM_SECTOR; ++sector) {
        for (int ring = 0; ring < PC_NUM_RING; ++ring) {
            if (total_count(ring, sector) <= 0) {
                continue;
            }

            const double dynamic_ratio =
                dynamic_count(ring, sector) / total_count(ring, sector);

            if (dynamic_ratio > DYNAMIC_RATIO_THRESH &&
                dynamic_count(ring, sector) > 5) {
                dynamic_mask(ring, sector) = 0;
            }
        }
    }

    return dynamic_mask;
}

/**
 * @brief 改进的Ring key计算函数，对动态遮挡不敏感
 * @param _desc Scan Context矩阵
 * @param _dynamic_mask 动态标签矩阵
 * @return 改进的Ring key
 */
Eigen::MatrixXd SCManager::makeRobustRingkeyFromScancontext(
    Eigen::MatrixXd &_desc, 
    Eigen::MatrixXd &_dynamic_mask) {
    
    Eigen::MatrixXd robust_ringkey(_desc.rows(), 1);
    
    for (int row_idx = 0; row_idx < _desc.rows(); row_idx++) {
        // 步骤1: 收集未被遮挡的观测值
        std::vector<double> valid_observations;
        
        for (int col_idx = 0; col_idx < _desc.cols(); col_idx++) {
            if (_dynamic_mask(row_idx, col_idx) > 0.5) {  // 可观测
                double obs_value = _desc(row_idx, col_idx);
                if (obs_value != 0) {  // 有效观测值
                    valid_observations.push_back(obs_value);
                }
            }
        }
        
        if (valid_observations.empty()) {
            // 如果该环没有有效观测，使用默认值
            robust_ringkey(row_idx, 0) = 0.0;
            continue;
        }
        
        // 步骤2: 计算中位数
        double median_value = 0.0;
        std::sort(valid_observations.begin(), valid_observations.end());
        int n = valid_observations.size();
        
        if (n % 2 == 1) {
            // 奇数个元素
            median_value = valid_observations[n / 2];
        } else {
            // 偶数个元素，取中间两个的平均
            median_value = (valid_observations[n / 2 - 1] + valid_observations[n / 2]) / 2.0;
        }
        
        // 步骤3: 计算权重（可观测比例）
        double static_sectors = 0.0;
        for (int col_idx = 0; col_idx < _desc.cols(); col_idx++) {
            if (_dynamic_mask(row_idx, col_idx) > 0.5) {
                static_sectors += 1.0;
            }
        }
        double weight = static_sectors / _desc.cols();
        
        // Ring key = 权重 * 中位数
        robust_ringkey(row_idx, 0) = median_value;
    }
    
    return robust_ringkey;
}

/**
 * @brief 基于动态标签加权的Scan Context距离计算
 * @param _sc1 第一个Scan Context
 * @param _sc2 第二个Scan Context
 * @param _dynamic_mask1 第一个动态标签矩阵
 * @param _dynamic_mask2 第二个动态标签矩阵
 * @return 加权距离和最佳旋转对齐
 */
std::pair<double, int> SCManager::distanceBtnScanContextWithDynamic(
    MatrixXd &_sc1, MatrixXd &_sc2,
    MatrixXd &_dynamic_mask1, MatrixXd &_dynamic_mask2) {
    
    // 1. 快速对齐使用原有的扇区键（保持不变）
    MatrixXd vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
    MatrixXd vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
    int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);
    
    const int SEARCH_RADIUS = round(0.5 * SEARCH_RATIO * _sc1.cols());
    std::vector<int> shift_idx_search_space{argmin_vkey_shift};
    for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++) {
        shift_idx_search_space.push_back((argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols());
        shift_idx_search_space.push_back((argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols());
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());
    
    // 2. 使用加权的列相似度计算距离
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    
    for (int num_shift : shift_idx_search_space) {
        // 旋转第二个Scan Context和对应的动态标签矩阵
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);
        MatrixXd mask2_shifted = circshift(_dynamic_mask2, num_shift);
        
        // 计算加权余弦距离
        double cur_sc_dist = weightedDistDirectSC(_sc1, sc2_shifted, _dynamic_mask1, mask2_shifted);
        
        if (cur_sc_dist < min_sc_dist) {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }
    
    return make_pair(min_sc_dist, argmin_shift);
}

/**
 * @brief 计算单列的可观测性
 * @param _col 列向量
 * @param _mask_col 列的动态标签
 * @return 可观测权重 [0, 1]
 */
double SCManager::computeColumnObservability(const VectorXd &_col, const VectorXd &_mask_col) {
    int total_elements = _col.size();
    int static_elements = 0;
    int valid_elements = 0;
    
    for (int i = 0; i < total_elements; i++) {
        if (_mask_col(i) > 0.5) {  // 静态区域
            static_elements++;
            valid_elements++;
        }
    }
    
    if (total_elements == 0) {
        return 0.0;
    }
    
    // 可观测性 = 静态区域比例 * 有效观测比例
    double static_ratio = static_cast<double>(static_elements) / total_elements;
    double valid_ratio = (static_elements > 0) ? 
        static_cast<double>(valid_elements) / static_elements : 0.0;
    
    return static_ratio * valid_ratio;
}

/**
 * @brief 基于动态标签的加权列相似度计算
 * @param _sc1 第一个Scan Context
 * @param _sc2 第二个Scan Context
 * @param _mask1 第一个动态标签矩阵
 * @param _mask2 第二个动态标签矩阵
 * @return 加权距离
 */
double SCManager::weightedDistDirectSC(MatrixXd &_sc1, MatrixXd &_sc2,
                                       MatrixXd &_mask1, MatrixXd &_mask2) {
    
    int num_valid_cols = 0;  // 有效的列数（两列都有足够的可观测性）
    double weighted_sector_similarity_sum = 0.0;
    
    for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++) {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);
        VectorXd mask_col1 = _mask1.col(col_idx);
        VectorXd mask_col2 = _mask2.col(col_idx);
        
        // 计算该列在两帧中的可观测权重
        // double weight1 = computeColumnObservability(col_sc1, mask_col1);
        // double weight2 = computeColumnObservability(col_sc2, mask_col2);
        
        // 如果任一列可观测性太低，跳过
        // if (weight1 < 0.3 || weight2 < 0.3) {
        //     continue;
        // }
        
        // 计算列相似度
        double col_similarity = 0.0;
        int valid_elements = 0;
        
        for (int row_idx = 0; row_idx < _sc1.rows(); row_idx++) {
            // 只考虑两帧中都可观测的元素
            if (_mask1(row_idx, col_idx) > 0.5 && _mask2(row_idx, col_idx) > 0.5) {
                
                valid_elements++;
            }
        }
        
        // 如果有效元素太少，跳过
        if (valid_elements < 2) {
            continue;
        }
        
        // 计算余弦相似度
        VectorXd valid_sc1 = VectorXd::Zero(valid_elements);
        VectorXd valid_sc2 = VectorXd::Zero(valid_elements);
        int idx = 0;
        
        for (int row_idx = 0; row_idx < _sc1.rows(); row_idx++) {
            if (_mask1(row_idx, col_idx) > 0.5 && _mask2(row_idx, col_idx) > 0.5) {
                double val1 = _sc1(row_idx, col_idx);
                double val2 = _sc2(row_idx, col_idx);
                valid_sc1(idx) = val1;
                valid_sc2(idx) = val2;
                idx++;
            }
        }
        
        if (valid_sc1.norm() > 1e-6 && valid_sc2.norm() > 1e-6) {
            col_similarity = valid_sc1.dot(valid_sc2) / (valid_sc1.norm() * valid_sc2.norm());
        } else {
            continue;  // 无效列
        }
        
        // 使用权重计算加权相似度
        // double weight = (weight1 + weight2) / 2.0;  // 平均权重
        // weighted_sector_similarity_sum += weight * col_similarity;
        weighted_sector_similarity_sum += col_similarity;
        num_valid_cols++;
    }
    
    if (num_valid_cols == 0) {
        return 1.0;  // 最大距离
    }
    
    double weighted_sc_sim = weighted_sector_similarity_sum / num_valid_cols;
    return 1.0 - weighted_sc_sim;
}

/**
 * @brief 完整的动态鲁棒闭环检测流程
 */
std::pair<int, float> SCManager::detectLoopClosureIDWithDynamic() {
    int loop_id = -1;
    top_candidates_.clear();
    
    if (polarcontext_invkeys_mat_.size() < NUM_EXCLUDE_RECENT + 1) {
        return {loop_id, 0.0};
    }
    
    // 获取当前帧信息
    auto curr_key = polarcontext_invkeys_mat_.back();
    auto curr_desc = polarcontexts_.back();
    auto curr_dynamic_mask = dynamic_masks_.back();
    
    // 重建KD树（使用改进的Ring key）
    if (tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) {
        TicTocV2 t_tree_construction;
        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.assign(
            polarcontext_invkeys_mat_.begin(), 
            polarcontext_invkeys_mat_.end() - NUM_EXCLUDE_RECENT
        );
        
        polarcontext_tree_.reset();
        polarcontext_tree_ = std::make_unique<InvKeyTree>(
            PC_NUM_RING, polarcontext_invkeys_to_search_, 10
        );
    }
    tree_making_period_conter++;
    
    // 步骤1: 从KD树中搜索候选
    std::vector<size_t> candidate_indexes(NUM_CANDIDATES_FROM_TREE);
    std::vector<float> out_dists_sqr(NUM_CANDIDATES_FROM_TREE);
    
    nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_FROM_TREE);
    knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);
    polarcontext_tree_->index->findNeighbors(
        knnsearch_result, &curr_key[0], nanoflann::SearchParams(10)
    );

    for (int i = 0; i < NUM_CANDIDATES_FROM_TREE; ++i) {
        top_candidates_.emplace_back(
            static_cast<int>(candidate_indexes[i]),
            out_dists_sqr[i]
        );
    }
    
    // 步骤2: 使用加权Scan Context距离进行精细匹配
    double min_sc_dist = 10000000;
    int nn_align = 0;
    int nn_idx = 0;
    
    for (int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++) {
        int candidate_idx = candidate_indexes[candidate_iter_idx];
        MatrixXd polarcontext_candidate = polarcontexts_[candidate_idx];
        MatrixXd dynamic_mask_candidate = dynamic_masks_[candidate_idx];
        
        // 使用加权Scan Context距离
        std::pair<double, int> sc_dist_result = distanceBtnScanContextWithDynamic(
            curr_desc, polarcontext_candidate,
            curr_dynamic_mask, dynamic_mask_candidate
        );
        
        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;
        
        if (candidate_dist < min_sc_dist) {
            min_sc_dist = candidate_dist;
            nn_align = candidate_align;
            nn_idx = candidate_idx;
        }
    }
    
    // 步骤3: 阈值判断
    if (min_sc_dist < SC_DIST_THRES) {
        loop_id = nn_idx;
        std::cout << "[Loop found] Weighted SC distance: " << min_sc_dist 
                  << " between " << polarcontexts_.size()-1 << " and " << nn_idx << std::endl;
    } else {
        std::cout << "[Not loop] Weighted SC distance: " << min_sc_dist 
                  << " between " << polarcontexts_.size()-1 << " and " << nn_idx << std::endl;
    }
    
    float yaw_diff_rad = deg2rad(nn_align * PC_UNIT_SECTORANGLE);
    return {loop_id, min_sc_dist};
}

const std::vector<std::pair<int, float>>& SCManager::getTopCandidates() const {
    return top_candidates_;
}

/**
 * @brief 保存Scan Context和动态标签
 */
void SCManager::makeAndSaveScancontextAndKeysWithDynamic(
    pcl::PointCloud<SCPointType> &_scan_down,
    pcl::PointCloud<SCPointType> &_doppler_cloud) {
    
    // 生成Scan Context
    Eigen::MatrixXd sc;
    if (SC_CHANNEL == 1){
      sc = makeScancontext(_scan_down);  // v1
    } 
    if (SC_CHANNEL == 2){
      sc = makeScancontext_intensity(_scan_down);  // v2
    }
    // std::cout << "sc:" << sc << std::endl;
    // 生成动态标签矩阵
    Eigen::MatrixXd dynamic_mask = makeDynamicMaskFromDopplerCloud(_doppler_cloud, _scan_down);
    // std::cout << "dynamic mask:" << dynamic_mask << std::endl;
    // 使用改进的Ring key计算方法
    Eigen::MatrixXd ringkey = makeRobustRingkeyFromScancontext(sc, dynamic_mask);
    
    // 扇区键计算方式不变
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext(sc);
    
    // 转换格式
    std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);
    
    // 保存
    polarcontexts_.push_back(sc);
    polarcontext_invkeys_.push_back(ringkey);
    polarcontext_vkeys_.push_back(sectorkey);
    polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);
    
    // 保存动态标签矩阵
    dynamic_masks_.push_back(dynamic_mask);
}

void SCManager::CancelScancontextAndKeysWithDynamic() {
    // 移除最近添加的 Scan Context
    if (!polarcontexts_.empty()) {
        polarcontexts_.pop_back();
    }
    // 移除对应的鲁棒 Ring Key（矩阵形式）
    if (!polarcontext_invkeys_.empty()) {
        polarcontext_invkeys_.pop_back();
    }
    // 移除对应的 Sector Key
    if (!polarcontext_vkeys_.empty()) {
        polarcontext_vkeys_.pop_back();
    }
    // 移除对应的 Ring Key（向量形式，用于 KD-tree 检索）
    if (!polarcontext_invkeys_mat_.empty()) {
        polarcontext_invkeys_mat_.pop_back();
    }
    // 移除对应的动态标签矩阵
    if (!dynamic_masks_.empty()) {
        dynamic_masks_.pop_back();
    }
}
