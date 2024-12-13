_base_ = './cascade_mask_rcnn_r18_fpn_1x_nuim.py'

# learning policy
lr_config = dict(step=[16, 19])
runner = dict(max_epochs=20)

load_from = 'ckpts/resnet18-imagenet-coco.pth'  # noqa
