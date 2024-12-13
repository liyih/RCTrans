_base_ = [
    '../_base_/models/cascade_mask_rcnn_r18_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]

# learning policy
lr_config = dict(step=[16, 19])
runner = dict(max_epochs=20)

load_from = None  # noqa