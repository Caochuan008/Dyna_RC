# Dataset release fields

Fill this page when the downloadable data are ready. The ROS package accepts
local files and does not download bags automatically.

| Pair | Bag URL | Map/GT URL | SHA-256 | Query interval | License/citation |
|---|---|---|---|---|---|
| NTU-LOOP1 | TODO | TODO | TODO | TODO | TODO |
| SJTU-IMR2 | TODO | TODO | TODO | TODO | TODO |
| SJTU-ROAD/DYNA | TODO | TODO | TODO | TODO | TODO |
| NTU-CP/SMOKE | TODO | TODO | TODO | TODO | TODO |

For each release, also state:

- Whether the bag contains raw radar messages or preprocessed static/dynamic streams.
- Raw topic name, ROS message type, radar model, frame ID, units, and channel mapping.
- The matching map pose file and numbered PCD directory, including dynamic map clouds if available.
- Map/query feature channel, range, z filter, and preprocessing calibration.
- GT timestamp convention, the mapping coordinate frame, and any cross-session alignment applied.
- Whether the download is mirrored or links to the original data provider.

For first/second traversals reconstructed together, retain their shared world
frame. For separate sessions, provide the query reference already transformed
into the mapping frame and document the calibrated sensor transforms used.
Do not describe an unaligned query trajectory as map-frame ground truth.
