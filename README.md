

## 1、add offline run mode for visulization

### a. run mode inference on remote server

```shell
python3 projects/BEVFusion/demo/multi_modality_demo.py   \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin   \
    demo/data/nuscenes/         \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl          \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py    \
    checkpoint/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth      \
    --cam-type all              \
    --score-thr 0.2           \
    --save-pred
```

### b. run visulization on PC
```shell
python3 projects/BEVFusion/demo/multi_modality_demo.py   \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin   \
    demo/data/nuscenes/         \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl          \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py    \
    checkpoint/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth      \
    --cam-type all              \
    --score-thr 0.2           \
    --load-pred                 \
    --show
```

## 2、add batch test for nuscenes dataset
```shell
python projects/BEVFusion/demo/multi_modality_demo_batch.py \
    data/nuscenes/nuscenes_infos_test.pkl \
    data/nuscenes/samples \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    checkpoint/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth \
    --score-thr 0.2
```