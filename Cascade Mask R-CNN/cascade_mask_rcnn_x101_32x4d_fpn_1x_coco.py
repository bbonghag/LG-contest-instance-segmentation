_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32, # 원래 32
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))


# log 저장 위치
checkpoint_config = dict(interval=1,out_dir='/content/drive/MyDrive/dataset/lg/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco')

# 사전 가중치 사용
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth'