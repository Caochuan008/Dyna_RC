#pragma once

#include "scancontext/tic_toc.h"
#include "scancontext/KDTreeVectorOfVectorsAdaptor.h"
#include "scancontext/nanoflann.hpp"
// #include "ravoslam/radar_point_type.h"

using namespace nanoflann;

using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

using SCPointType = pcl::PointXYZI;
// using SCPointType = PointXYZRGBCov;
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

class SCManager {
 public:
  SCManager() = default;  // reserving data space (of std::vector) could be considered.
                          // but the descriptor is lightweight so don't care.

  Eigen::MatrixXd makeScancontext(pcl::PointCloud<SCPointType> &_scan_down);
  Eigen::MatrixXd makeScancontext_intensity(pcl::PointCloud<SCPointType> &_scan_down);
  Eigen::MatrixXd makeRingkeyFromScancontext(Eigen::MatrixXd &_desc);
  Eigen::MatrixXd makeSectorkeyFromScancontext(Eigen::MatrixXd &_desc);

  int fastAlignUsingVkey(MatrixXd &_vkey1, MatrixXd &_vkey2);
  double distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2);  // "d" (eq 5) in the original paper (IROS 18)
  std::pair<double, int> distanceBtnScanContext(MatrixXd &_sc1, MatrixXd &_sc2);  // "D" (eq 6) in the original paper (IROS 18)

  // User-side API
  void makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> &_scan_down);
  std::pair<int, float> detectLoopClosureID(void);  // int: nearest node index, float: relative yaw

// 新添加的函数声明
  Eigen::MatrixXd makeDynamicMaskFromDopplerCloud(
        pcl::PointCloud<SCPointType> &_doppler_cloud,
        pcl::PointCloud<SCPointType> &_scan_down);
    
  Eigen::MatrixXd makeRobustRingkeyFromScancontext(
        Eigen::MatrixXd &_desc, 
        Eigen::MatrixXd &_dynamic_mask);
    
  double weightedDistDirectSC(
        MatrixXd &_sc1, MatrixXd &_sc2,
        MatrixXd &_mask1, MatrixXd &_mask2);
    
  double computeColumnObservability(const VectorXd &_col, const VectorXd &_mask_col);
    
  std::pair<double, int> distanceBtnScanContextWithDynamic(
        MatrixXd &_sc1, MatrixXd &_sc2,
        MatrixXd &_dynamic_mask1, MatrixXd &_dynamic_mask2);
    
  void makeAndSaveScancontextAndKeysWithDynamic(
        pcl::PointCloud<SCPointType> &_scan_down,
        pcl::PointCloud<SCPointType> &_doppler_cloud);

  void CancelScancontextAndKeysWithDynamic();
    
  std::pair<int, float> detectLoopClosureIDWithDynamic();

  const std::vector<std::pair<int, float>>& getTopCandidates() const;
  std::vector<std::pair<int, float>> top_candidates_;

  // for ltslam
  // User-side API for multi-session
  void saveScancontextAndKeys(Eigen::MatrixXd _scd);
  std::pair<int, float> detectLoopClosureIDBetweenSession(
      std::vector<float> &curr_key, Eigen::MatrixXd &curr_desc);

  inline const Eigen::MatrixXd& getConstRefRecentSCD(void) {
    return polarcontexts_.back();
  }

 public:
  // hyper parameters ()
  const double LIDAR_HEIGHT = 0.6;
            // lidar height : add this for simply directly using lidar scan in
            // the lidar local coord (not robot base coord) / if you use
            // robot-coord-transformed lidar scans, just set this as 0.

  const int PC_NUM_RING = 20;    // 20 in the original paper (IROS 18)
  const int PC_NUM_SECTOR = 20;  // 60 in the original paper (IROS 18) for radar the azimuth range is 1/3 of lidar
  double PC_MAX_RADIUS = 80.0;  // 80 meter max in the original paper (IROS 18)
  const double PC_UNIT_SECTORANGLE = 120.0 / double(PC_NUM_SECTOR); //radar azimuth range is 120
  const double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);
  int SC_CHANNEL = 1;

  // tree
  const int NUM_EXCLUDE_RECENT = 50;
          // simply just keyframe gap (related with loopClosureFrequency in
          // yaml), but node position distance-based exclusion is ok.
  const int NUM_CANDIDATES_FROM_TREE = 5;  // 10 is enough. (refer the IROS 18 paper)

  // loop thres
  const double SEARCH_RATIO = 0.1;
            // for fast comparison, no Brute-force, but search 10 % is okay. //
            // not was in the original conf paper, but improved ver.
  // const double SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare
            // false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score
            // check (e.g., in LeGO-LOAM) should be required for robustness)

  double SC_DIST_THRES = 0.2;  // 0.4-0.6 is good choice for using with robust kernel (e.g.,
            // Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

  // config
  const int TREE_MAKING_PERIOD_ = 80;
          // i.e., remaking tree frequency, to avoid non-mandatory every
          // remaking, to save time cost / in the LeGO-LOAM integration, it is
          // synchronized with the loop detection callback (which is 1Hz) so it
          // means the tree is updated evrey 10 sec. But you can use the
          // smaller value because it is enough fast ~ 5-50ms wrt N.
  int tree_making_period_conter = 0;

  // setter
  void setSCdistThres(double _new_thres);
  void setMaximumRadius(double _max_r);
  void setScChannel(int _sc_channel);
  

  // data
  std::vector<double> polarcontexts_timestamp_;  // optional.
  std::vector<Eigen::MatrixXd> polarcontexts_;
  std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
  std::vector<Eigen::MatrixXd> polarcontext_vkeys_;
  std::vector<Eigen::MatrixXd> dynamic_masks_;  // 存储动态标签矩阵

  KeyMat polarcontext_invkeys_mat_;
  KeyMat polarcontext_invkeys_to_search_;
  std::unique_ptr<InvKeyTree> polarcontext_tree_;

  bool is_tree_batch_made = false;
  std::unique_ptr<InvKeyTree> polarcontext_tree_batch_;

};  // SCManager
