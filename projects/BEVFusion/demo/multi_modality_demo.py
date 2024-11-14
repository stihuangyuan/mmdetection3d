# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

from mmdet3d.apis import inference_multi_modality_detector, init_model
from mmdet3d.registry import VISUALIZERS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('img', help='image file')
    parser.add_argument('ann', help='ann file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cam-type',
        type=str,
        default='CAM_FRONT',
        help='choose camera type to inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--save-pred',
        action='store_true',
        help='save model inference')
    parser.add_argument(
        '--load-pred',
        action='store_true',
        help='load model inference')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()
    return args


def main(args):
    if args.load_pred:
        cfg_visualizer = {'type': 'Det3DLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}], 'name': 'visualizer'}
        dataset_meta = {'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], 
                        'version': 'v1.0-trainval', 
                        'CLASSES': ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'), 
                        'DATASET': 'Nuscenes', 
                        'palette': [(255, 158, 0), (255, 99, 71), (255, 140, 0), (255, 127, 80), (233, 150, 70), (220, 20, 60), (255, 61, 99), (0, 0, 250), (47, 79, 79), (112, 128, 144)]}
        visualizer = VISUALIZERS.build(cfg_visualizer)
        visualizer.dataset_meta = dataset_meta
        
        import pickle
        with open('test.pkl', 'rb') as f:
            data = pickle.load(f)
            result, points = data['result'], data['points']

        import torch
        from mmdet3d.structures import LiDARInstance3DBoxes
        result.pred_instances_3d.scores_3d = torch.from_numpy(result.pred_instances_3d.scores_3d)
        result.pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(result.pred_instances_3d.bboxes_3d, box_dim=9)
        result.pred_instances_3d.labels_3d = torch.from_numpy(result.pred_instances_3d.labels_3d)
    
    else:
        # build the model from a config file and a checkpoint file
        model = init_model(args.config, args.checkpoint, device=args.device)

        # init visualizer
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta

        # test a single image and point cloud sample
        import time
        t1 = time.time()
        result, data = inference_multi_modality_detector(model, args.pcd, args.img,
                                                        args.ann, args.cam_type)
        t2 = time.time()
        print('infer time cost: ', t2 -t1)

        points = data['inputs']['points']

    if args.save_pred:
        # convert tensor to nmpy to save
        points_np = data['inputs']['points'].cpu().numpy()
        import copy
        result_np = copy.deepcopy(result)
        result_np.pred_instances_3d.scores_3d = result_np.pred_instances_3d.scores_3d.cpu().numpy()
        result_np.pred_instances_3d.bboxes_3d = result_np.pred_instances_3d.bboxes_3d.cpu().numpy()
        result_np.pred_instances_3d.labels_3d = result_np.pred_instances_3d.labels_3d.cpu().numpy()
        import pickle
        with open('test.pkl', 'wb') as f:
            pickle.dump({'result': result_np, 'points': points_np}, f)

    if isinstance(result.img_path, list):
        img = []
        for img_path in result.img_path:
            single_img = mmcv.imread(img_path)
            single_img = mmcv.imconvert(single_img, 'bgr', 'rgb')
            img.append(single_img)
    else:
        img = mmcv.imread(result.img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
    data_input = dict(points=points, img=img)

    # show the results
    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=args.show,
        wait_time=-1,
        out_file=args.out_dir,
        pred_score_thr=args.score_thr,
        vis_task='multi-modality_det')


if __name__ == '__main__':
    args = parse_args()
    main(args)
