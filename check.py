# import torch
# import pdb
# import pickle
# # lidar_pre = torch.load('/mnt/share_disk/lyh/RCDETR/cmt_ema.pth')['state_dict'] 

# with open('/mnt/share_disk/lyh/RCDETR/data/nuscenes_radar_temporal_infos_train.pkl', 'rb') as f:
#     infos_train = pickle.load(f)

# with open('/mnt/share_disk/lyh/RCDETR/data/nuscenes_radar_temporal_infos_val.pkl', 'rb') as f:
#     infos_val = pickle.load(f)

# infos_trainval = {}
# infos_trainval['metadata'] = infos_train['metadata']
# infos_trainval['infos'] = infos_train['infos'] + infos_val['infos']
# pdb.set_trace()
# with open('/mnt/share_disk/lyh/RCDETR/data/nuscenes_radar_temporal_infos_trainval.pkl', 'wb') as f:
#     pickle.dump(infos_trainval, f) 
# pdb.set_trace()

import torch

pretrain_dict = torch.load('ckpts/eva02_L_coco_det_sys_o365.pth', map_location=torch.device('cpu'))
pretrain_dict = pretrain_dict["model"]
print(pretrain_dict.keys())
remapped_dict = {}
for k,v in pretrain_dict.items():
    if "backbone.net" in k:
        remapped_dict[k.replace("backbone.net.", "img_backbone.")] = v
    if "backbone.simfp" in k:
        remapped_dict[k.replace("backbone.", "img_backbone.adapter.")] = v
torch.save(remapped_dict,'ckpts/eva02_L_coco_det_sys_o365_remapped.pth')