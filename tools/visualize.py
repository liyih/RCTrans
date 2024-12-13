import os
import tqdm
import json
from visual_nuscenes import NuScenes
use_gt = False
out_dir = './result_vis/swint/'
result_json = "test/rcdetr_90e_256×704_swinT/Tue_Aug__6_13_27_00_2024/pts_bbox/results_nusc"
dataroot='./data/nuscenes'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if use_gt:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = False, annotations = "sample_annotation")
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True, pred = True, annotations = result_json, score_thr=0.25)

with open('{}.json'.format(result_json)) as f:
    table = json.load(f)
tokens = list(table['results'].keys())
index=0
for token in tqdm.tqdm(tokens[:300]):
    index += 1
    if use_gt:
        nusc.render_sample(token, out_path = out_dir+str(index)+"_gt.png", verbose=False)
    else:
        nusc.render_sample(token, out_path = out_dir+str(index)+"_pred.png", verbose=False)

