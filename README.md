<div align="center">
<h1>RCTrans</h1>
<h3>[AAAI 2025] RCTrans: Radar-Camera Transformer via Radar Densiffer and Sequential Decoder for 3D Object Detection</h3>
<h4>Yiheng Li, Yang Yang and Zhen Lei<h4>
<h5>CBSR&CASIA<h5>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

## Introduction

This repository is an official implementation of RCTrans.

## News
- [2024/12/13] Codes and weights are released.
- [2024/12/10] RCTrans is accepted by AAAI 2023 🎉🎉.

## Environment Setting
```
conda create -n RCTrans python=3.8
conda activate RCTrans
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install flash-attn==0.2.2 --no-build-isolation
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
cd mmdetection3d
pip install -v -e .
cd ..
pip install ipython
pip install fvcore
pip install spconv-cu111==2.1.21
pip install yapf==0.40.0
pip install setuptools==59.5.0
pip install ccimport==0.3.7
pip install pccm==0.3.4
pip install timm
```
## Data Preparation
```
python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data --extra-tag nuscenes_radar --version v1.0
```
Folder structure
```
RCTrans
├── projects/
├── mmdetection3d/
├── nusc_tracking/
├── tools/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   ├── nuscenes_radar_temporal_infos_train.pkl
|   ├── nuscenes_radar_temporal_infos_val.pkl
|   ├── nuscenes_radar_temporal_infos_test.pkl
```
Or you can directly use our pre-generated pickles here. [Val](https://drive.usercontent.google.com/download?id=1CLs4zi2tmkBl33XzEkvmUDT9an-2N9c5&export=download&authuser=0&confirm=t&uuid=22c1cee9-3b91-4b7f-84b8-fd69aae10224&at=APvzH3oFQ5HqwWzKXsSTckzGP1gP:1734076238954)[Train](https://drive.usercontent.google.com/download?id=1m2rggU4jzuBPDPfCbC3u0G5ugD-e8P9t&export=download&authuser=0&confirm=t&uuid=61169d3e-e31b-4ad7-920c-3a746eceba74&at=APvzH3qPOu74S9o-v19hxWgZU-ku:1734076306697)[Test](https://drive.usercontent.google.com/download?id=1Xhc1DMbi67YsV7nis26GWOjxjVAmTF3o&export=download&authuser=0&confirm=t&uuid=86051653-5de3-4383-ab97-ab43f0ec93d1&at=APvzH3p-l9SdhykVspp5eDGxmLMa:1734076308824)