# 12epoch
_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'

# 20epoch
# _base_ = './cascade_mask_rcnn_r50_fpn_20e_coco.py'

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

# log 저장 위치
checkpoint_config = dict(interval=1,out_dir='/content/drive/MyDrive/dataset/lg/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco')

# 사전 가중치 사용 - 12epoch
load_from = '/content/content/lg_test/mmdetection/checkpoint/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth'

# 사전 가중치 - 20epoch
# load_from = '/content/content/lg_test/mmdetection/checkpoint/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'

