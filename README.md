<div align="center">
<h1>RCTrans</h1>
<h3>[AAAI 2025] RCTrans: Radar-Camera Transformer via Radar Densiffer and Sequential Decoder for 3D Object Detection</h3>
<h4>Yiheng Li, Yang Yang and Zhen Lei<h4>
<h5>CASIA<h5>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()

## Introduction

This repository is an official implementation of RCTrans.

## News
- [2024/12/13] Codes and weights are released.
- [2024/12/10] RCTrans is accepted by AAAI 2023 🎉🎉.

## Environment setting
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