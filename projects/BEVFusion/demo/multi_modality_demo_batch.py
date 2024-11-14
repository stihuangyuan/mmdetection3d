# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

from mmdet3d.apis import inference_multi_modality_detector, init_model
from mmdet3d.registry import VISUALIZERS

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('pcd', help='Point cloud file')
    # parser.add_argument('img', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('root_folder', help='root data folder')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--cam-type',
    #     type=str,
    #     default='CAM_FRONT',
    #     help='choose camera type to inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args

import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import mmengine
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from typing import Optional, Sequence, Union
from collections import defaultdict

def process_ann_scenes(args):
    
    model = init_model(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    ann_file = args.ann
    root_path = args.root_folder

    ann_info = mmengine.load(ann_file)
    data_list = ann_info['data_list']
    
    cfg = model.cfg

    # build the data pipeline
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = \
        get_box_type(cfg.test_dataloader.dataset.box_type_3d)
    
    result_folder = './result_batchs'
    os.makedirs(result_folder, exist_ok=True)
    
    lidar_names = []
    cam_names = defaultdict(list)
    basename_2_indexs = defaultdict(list)
    for i in range(len(data_list)):
        data_info = data_list[i]
        lidar_path = data_info['lidar_points']['lidar_path']
        lidar_names.append(lidar_path)

        for cam_type, img_info in data_info['images'].items():
            cam_names[cam_type].append(img_info['img_path'])

        base_name = os.path.basename(lidar_path).split('_')[0]
        basename_2_indexs[base_name].append(i)
    print('total', len(basename_2_indexs), 'scenes')

    # import pdb; pdb.set_trace()
    for base_name, indexs in basename_2_indexs.items():
        videoWriter = None
        for i in tqdm(range(len(indexs)), desc='Processing {} ....'.format(base_name)):
            idx = indexs[i]
            data_info = data_list[idx]

            lidar_path = data_info['lidar_points']['lidar_path']
            lidar_path = os.path.join(root_path, 'LIDAR_TOP', lidar_path)
            assert os.path.exists(lidar_path), 'why {} not exit???'.format(lidar_path)

            cam_front_path = None
            img_paths = []
            for cam_type, img_info in data_info['images'].items(): #  dict_keys(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
                img_info['img_path'] = os.path.join(root_path, cam_type, img_info['img_path'])
                assert os.path.isfile(img_info['img_path']), f'{img_info["img_path"]} does not exist.'
                img_paths.append(img_info['img_path'])
                if cam_type == 'CAM_FRONT':
                    # print(img_info['img_path'])
                    cam_front_path = img_info['img_path']

            data_ = dict(
                lidar_points=dict(lidar_path=lidar_path),
                images=data_info['images'],
                box_type_3d=box_type_3d,
                box_mode_3d=box_mode_3d)

            if 'timestamp' in data_info:
                # Using multi-sweeps need `timestamp`
                data_['timestamp'] = data_info['timestamp']

            data_ = test_pipeline(data_)

            data = []
            data.append(data_)

            collate_data = pseudo_collate(data)
            
            t1 = time.time()
            with torch.no_grad():
                results = model.test_step(collate_data)
            t2 = time.time()
            print('infer time cost: ', t2 -t1)

            result, data = results[0], data[0]
            
            points = data['inputs']['points']

            if isinstance(img_paths, list):
                img = []
                for img_path in img_paths:
                    single_img = mmcv.imread(img_path)
                    single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
                    img.append(single_img)
            else:
                img = mmcv.imread(img_paths)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
            data_input = dict(points=points, img=img)

            # out_file = os.path.basename(cam_front_path)
            # out_file = os.path.join(result_folder, out_file[:-4] + '.png')

            # show the results
            vis_3d = visualizer.add_datasample(
                'result',
                data_input,
                data_sample=result,
                draw_gt=False,
                show=args.show,
                wait_time=-1,
                # out_file=out_file,
                pred_score_thr=args.score_thr,
                vis_task='multi-modality_det')
            
            vis_3d= np.ascontiguousarray(vis_3d)
            cv2.putText(vis_3d, str(i), (90, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if videoWriter is None:
                h, w = vis_3d.shape[:2]
                out_video_size = (w, h)
                videoWriter = cv2.VideoWriter('{}/demo-{}.mp4'.format(result_folder, base_name), 
                                    cv2.VideoWriter_fourcc(*'MP4V'), 30.0, out_video_size)
            videoWriter.write(vis_3d)

        videoWriter.release() 

if __name__ == '__main__':
    args = parse_args()
    process_ann_scenes(args)
