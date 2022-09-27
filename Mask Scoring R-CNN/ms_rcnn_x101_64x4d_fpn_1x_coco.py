_base_ = './ms_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))


# log 저장 위치 - 구글드라이브에 interval당 pth파일 저장.
checkpoint_config = dict(interval=1,out_dir='/content/drive/MyDrive/dataset/lg/ms_rcnn_x101_64x4d_fpn_1x_coco')

# 사전 가중치 사용(모델별 readme확인후 링크가져와서 사용.) - 12epoch
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth'


