#ifndef RAVOSLAM_POINT_TYPE_H
#define RAVOSLAM_POINT_TYPE_H
#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif
#include<pcl/point_types.h>
#include<pcl/point_cloud.h>
#include<pcl/register_point_struct.h>

struct RadarPointType {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY
  float range;
  float azimuth;
  float elevation;
  float velocity;
  float rcs;
  float flag;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

/**
 * @brief Construct a new point cloud register point struct object
 * 
 */
POINT_CLOUD_REGISTER_POINT_STRUCT (RadarPointType,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (float, range, range) (float, azimuth, azimuth) (float, elevation, elevation)
    (float, velocity, velocity) (float, rcs, rcs) (float, flag, flag)
    )

/**
 * @brief Pointcloud with rgb and brief descriptor
 * 
 */
struct RadarPointXYZRGBO {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY
  float r;
  float g;
  float b;
  // std::array<uint8_t, 32> orb_descriptor;
  std::uint32_t brief0;
  std::uint32_t brief1;
  std::uint32_t brief2;
  std::uint32_t brief3;
  std::uint32_t brief4;
  std::uint32_t brief5;
  std::uint32_t brief6;
  std::uint32_t brief7;
  // float velocity;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (RadarPointXYZRGBO,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (float, r, r) (float, g, g) (float, b, b) 
    // (std::array<uint8_t, 32>, orb_descriptor, orb_descriptor)
    (std::uint32_t, brief0, brief0) (std::uint32_t, brief1, brief1) (std::uint32_t, brief2, brief2) (std::uint32_t, brief3, brief3)
    (std::uint32_t, brief4, brief4) (std::uint32_t, brief5, brief5) (std::uint32_t, brief6, brief6) (std::uint32_t, brief7, brief7)
    // (float, velocity, velocity)
    )

/**
 * @brief Pointcloud with rgb and intensity
 * 
 */
struct PointXYZIRGB
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY
  PCL_ADD_RGB
  float color_cov_00;
  float color_cov_01;
  float color_cov_02;
  float color_cov_11;
  float color_cov_12;
  float color_cov_22;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// 注册自定义点类型
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGB,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, rgb, rgb)
    (float, color_cov_00, color_cov_00)
    (float, color_cov_01, color_cov_01)
    (float, color_cov_02, color_cov_02)
    (float, color_cov_11, color_cov_11)
    (float, color_cov_12, color_cov_12)
    (float, color_cov_22, color_cov_22)
);

struct PointXYZRGBCov
{
  PCL_ADD_POINT4D      // 继承 x,y,z 和 padding
  PCL_ADD_INTENSITY    // 继承 intensity
  PCL_ADD_RGB          // 继承 rgb
  float color_cov_00;
  float color_cov_01;
  float color_cov_02;
  float color_cov_11;
  float color_cov_12;
  float color_cov_22;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 默认构造函数（初始化协方差为单位矩阵）
  PointXYZRGBCov() : 
    color_cov_00(0.01f), color_cov_01(0.0f), color_cov_02(0.0f),
    color_cov_11(0.01f), color_cov_12(0.0f), color_cov_22(0.01f) 
  {}

  // 从 PointXYZRGB + 协方差矩阵构造
  PointXYZRGBCov(const pcl::PointXYZRGB& pt, 
                const Eigen::Matrix3f& cov = 0.01*Eigen::Matrix3f::Identity()) :
    x(pt.x), y(pt.y), z(pt.z),
    intensity(0.0f),
    rgb(pt.rgb),
    color_cov_00(cov(0,0)), 
    color_cov_01(cov(0,1)), 
    color_cov_02(cov(0,2)),
    color_cov_11(cov(1,1)), 
    color_cov_12(cov(1,2)), 
    color_cov_22(cov(2,2))
  {}

  // 从 PointXYZ + RGB + 协方差构造
  PointXYZRGBCov(const pcl::PointXYZ& pt, 
                const std::uint32_t& rgb_val,
                const Eigen::Matrix3f& cov = 0.01*Eigen::Matrix3f::Identity()) :
    x(pt.x), y(pt.y), z(pt.z),
    intensity(0.0f),
    rgb(rgb_val),
    color_cov_00(cov(0,0)), 
    color_cov_01(cov(0,1)), 
    color_cov_02(cov(0,2)),
    color_cov_11(cov(1,1)), 
    color_cov_12(cov(1,2)), 
    color_cov_22(cov(2,2))
  {}

  operator pcl::PointXYZRGB() const {
        pcl::PointXYZRGB pt;
        pt.x = x;
        pt.y = y;
        pt.z = z;
        pt.rgb = rgb;
        return pt;
    }
} EIGEN_ALIGN16;

// 注册自定义点类型
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBCov,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, rgb, rgb)
    (float, color_cov_00, color_cov_00)
    (float, color_cov_01, color_cov_01)
    (float, color_cov_02, color_cov_02)
    (float, color_cov_11, color_cov_11)
    (float, color_cov_12, color_cov_12)
    (float, color_cov_22, color_cov_22)
);


/**
 * @brief ZF radar point format
 * 
 */
struct ZFRadarPointType {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY
  float range;
  float azimuth;
  float elevation;
  float velocity;
  float snr;                   // SNR: Signal noice ratio (formerly intensity)
  float rcs;
  float detectionconfidence;   // Detection confidence of the target
  float mse_range;
  float mse_azimuth;
  float mse_elevation;
  float mse_velocity;
  // float mse_subarray;
  // float mse_subarray2nd_best;
  float std_dev_range;
  float std_dev_azimuthAngle;
  float std_dev_elevationAngle;
  float std_dev_velocity;
  float std_dev_rcs;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

/**
 * @brief Construct a new point cloud register point struct object
 * 
 */
POINT_CLOUD_REGISTER_POINT_STRUCT (ZFRadarPointType,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (float, range, range) (float, azimuth, azimuth) (float, elevation, elevation)
    (float, velocity, velocity) (float, snr, snr) (float, rcs, rcs)
    (float, detectionconfidence, detectionconfidence)
    (float, mse_range, mse_range) (float, mse_azimuth, mse_azimuth)
    (float, mse_elevation, mse_elevation) (float, mse_velocity, mse_velocity)
    (float, std_dev_range, std_dev_range) (float, std_dev_azimuthAngle, std_dev_azimuthAngle)
    (float, std_dev_elevationAngle, std_dev_elevationAngle) (float, std_dev_velocity, std_dev_velocity)
    (float, std_dev_rcs, std_dev_rcs)
    )

/**
 * @brief Oculii radar point format
 * 
 */
struct OculiiRadarPointType {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY
  float range;
  float azimuth;
  float elevation;
  float velocity;
  double rangeAccu;     /*!< Store the accuracy value of range data */
  double aziAccu;       /*!< Store the accuracy value of aziumith angle data */
  double eleAccu;       /*!< Store the accuracy value of elevation angle data */
  double dopplerAccu;   /*!< Store the accuracy value of doppler data */
  float recoveredSpeed; /*!< Target true speed segment along vehicle moving direction*/
  uint16_t dotFlags;              /*!< Point status flag */
  uint16_t denoiseFlag;           /*!< Mark of noise point */
  uint16_t historyFrameFlag;      /*!< Mark of the last frame detections */
  uint16_t dopplerCorrectionFlag;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

/**
 * @brief Construct a new point cloud register point struct object
 * 
 */
POINT_CLOUD_REGISTER_POINT_STRUCT(OculiiRadarPointType,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,intensity)
    (float, range, range)(float, azimuth, azimuth)(float, elevation, elevation)
    (float, velocity, velocity)
    (float, recoveredSpeed, recoveredSpeed)
    (double, rangeAccu, rangeAccu) (double, aziAccu, aziAccu)
    (double, eleAccu, eleAccu) (double, dopplerAccu, dopplerAccu)
    (uint16_t, dotFlags, dotFlags) (uint16_t, denoiseFlag, denoiseFlag)
    (uint16_t, historyFrameFlag, historyFrameFlag)
    (uint16_t, dopplerCorrectionFlag, dopplerCorrectionFlag)
    )

struct DeprecatedOculiiRadarPointType {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY
  float range;
  float azimuth;
  float elevation;
  float velocity;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

/**
 * @brief Construct a new point cloud register point struct object
 * 
 */
POINT_CLOUD_REGISTER_POINT_STRUCT(DeprecatedOculiiRadarPointType,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,intensity)
    (float, range, range)(float, azimuth, azimuth)(float, elevation, elevation)
    (float, velocity, velocity)
    )

/**
 * @brief Endeavour dataset radar point format
 * 
 */
struct EndeavourRadarPointType {
  PCL_ADD_POINT4D
  float vel_rad;
  float oritation;
  float range_var;
  float vel_rad_var;
  float ang_var;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

/**
 * @brief Construct a new point cloud register point struct object
 * 
 */
POINT_CLOUD_REGISTER_POINT_STRUCT(EndeavourRadarPointType,
    (float, x, x)(float, y, y)(float, z, z)(float, vel_rad, vel_rad)
    (float, oritation, oritation)(float, range_var, range_var)
    (float, vel_rad_var, vel_rad_var)(float, ang_var, ang_var)
    )

#endif // RAVOSLAM_POINT_TYPE_H
