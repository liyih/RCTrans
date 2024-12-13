#!/usr/bin/env bash
set -x
set +e
LOG_DIR=$1
nvidia-smi
pwd
ls /root/

mkdir data
ln -s /oss://haomo-algorithms/release/algorithms/manual_created_cards/628c5f667844160dda20fcc8 /root/worker/data/nuscenes

sleep 1s

cp /share/liyiheng/nuscenes_radar_temporal_infos_train.pkl /root/worker/data/
cp /share/liyiheng/nuscenes_radar_temporal_infos_test.pkl /root/worker/data/
cp /share/liyiheng/nuscenes_radar_temporal_infos_val.pkl /root/worker/data/
cp /share/liyiheng/nuscenes_radar_temporal_infos_trainval.pkl /root/worker/data/
echo "data init done"

mkdir ckpts

cp /share/liyiheng/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth /root/worker/ckpts/
cp /share/liyiheng/fcos3d_vovnet_imgbackbone-remapped.pth /root/worker/ckpts/
cp /share/liyiheng/swint-nuimages-pretrained-e2e.pth /root/worker/ckpts/
cp /share/liyiheng/resnet18-nuimages-pretrained-e2e.pth /root/worker/ckpts/
cp /share/liyiheng/eva02_L_coco_det_sys_o365_remapped.pth /root/worker/ckpts/
cp /share/liyiheng/dla34-nuimages-pretrained-e2e.pth /root/worker/ckpts/
cp /share/liyiheng/resume.pth /root/worker/ckpts/
echo "cpkts init done"

pip install flash-attn==0.2.2 --no-build-isolation
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
# git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
# git checkout v1.0.0rc6 
pip install -v -e .
cd /root/worker
# pip install mmdet3d==1.0.0rc6
pip install ipython
pip install fvcore
pip install spconv-cu111==2.1.21
pip install yapf==0.40.0
pip install setuptools==59.5.0
pip install ccimport==0.3.7
pip install pccm==0.3.4
pip install timm
cd /root/worker
pwd
pip list
echo "environment init done"

export PYTHONPATH=$PYTHONPATH:/root/worker

# bash tools/dist_train.sh projects/configs/RCDETR/rcdetr_90e_448×800_dla34.py 8 --work-dir ${LOG_DIR}
# bash tools/dist_train.sh projects/configs/RCDETR/rcdetr_90e_256×704_res50.py 8 --work-dir ${LOG_DIR}
# bash tools/dist_train.sh projects/configs/RCDETR/rcdetr_90e_256×704_res18.py 8 --work-dir ${LOG_DIR}
# bash tools/dist_train.sh projects/configs/RCDETR/rcdetr_90e_256×704_swinT.py 8 --work-dir ${LOG_DIR}
# bash tools/dist_train.sh projects/configs/RCDETR/rcdetr_90e_640×1600_trainval.py 8 --work-dir ${LOG_DIR}
bash tools/dist_train.sh projects/configs/RCDETR/rcdetr_60e_800×1600_vitl.py 8 --work-dir ${LOG_DIR}
# bash tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py 8 --work-dir ${LOG_DIR}