# Third-party source inventory

| Component | Files | Source / notice |
|---|---|---|
| Scan Context-derived descriptor code | `src/Scancontext.cpp`, `include/scancontext/` | Modified implementation supplied with the research package; [original Scan Context project](https://github.com/irapkaist/scancontext). The supplied snapshot does not contain a separate Scan Context license file. Preserve provenance and confirm the applicable upstream terms before release. |
| nanoflann 1.3.2 | `include/scancontext/nanoflann.hpp`, `KDTreeVectorOfVectorsAdaptor.h` | BSD notices remain in the headers; version macro `0x132`. [Upstream](https://github.com/jlblancoc/nanoflann). |
| RaPlace-Light C++ | `third_party/raplace/` | [Upstream](https://github.com/hyesu-jang/RaPlace); original hashes and local changes in [SOURCE.txt](third_party/raplace/SOURCE.txt). No license file was included in the supplied archive. |
| ReFeree adaptation | `third_party/referee/` | [SPARO ReFeree](https://github.com/sparolab/ReFeree); supplied BSD-3-Clause text preserved in [LICENSE](third_party/referee/LICENSE). The supplied upstream README also requests author contact for commercial use. |

The RaPlace and ReFeree ROS adapters rasterize 4D point clouds. They are adaptations
for this evaluation pipeline, not native spinning-radar-image reproductions.
Project packaging does not change ownership or relicense any vendored component.
