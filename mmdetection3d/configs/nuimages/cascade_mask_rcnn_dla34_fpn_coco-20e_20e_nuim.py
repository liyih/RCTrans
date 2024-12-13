_base_ = './cascade_mask_rcnn_dla34_fpn_1x_nuim.py'

# learning policy
lr_config = dict(step=[16, 19])
runner = dict(max_epochs=20)

# load_from = 'ckpts/dla34-imagenet-coco.pth'  # noqa
