<div align="center">
<!-- <h1>RCTrans</h1> -->
<h3>[AAAI 2025] RCTrans: Radar-Camera Transformer via Radar Densifier and Sequential Decoder for 3D Object Detection</h3>
<h4>Yiheng Li, Yang Yang and Zhen Lei<h4>
<h5>MAIS&CASIA, UCAS<h5>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.12799)

## Introduction

This repository is an official implementation of RCTrans.

## News
- [2024/12/18] Camera Ready version is released.
- [2024/12/13] Codes and weights are released.
- [2024/12/10] RCTrans is accepted by AAAI 2025 🎉🎉.

## Environment Setting
```
conda create -n RCTrans python=3.8
conda activate RCTrans
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
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
Or you can directly use our pre-generated pickles here. [Val](https://drive.usercontent.google.com/download?id=1CLs4zi2tmkBl33XzEkvmUDT9an-2N9c5&export=download&authuser=0&confirm=t&uuid=22c1cee9-3b91-4b7f-84b8-fd69aae10224&at=APvzH3oFQ5HqwWzKXsSTckzGP1gP:1734076238954) [Train](https://drive.usercontent.google.com/download?id=1m2rggU4jzuBPDPfCbC3u0G5ugD-e8P9t&export=download&authuser=0&confirm=t&uuid=61169d3e-e31b-4ad7-920c-3a746eceba74&at=APvzH3qPOu74S9o-v19hxWgZU-ku:1734076306697) [Test](https://drive.usercontent.google.com/download?id=1Xhc1DMbi67YsV7nis26GWOjxjVAmTF3o&export=download&authuser=0&confirm=t&uuid=86051653-5de3-4383-ab97-ab43f0ec93d1&at=APvzH3p-l9SdhykVspp5eDGxmLMa:1734076308824)
## Train & Inference
Train
```
export PYTHONPATH=$PYTHONPATH:/xxx/xxx/RCTrans/
bash tools/dist_train.sh projects/configs/RCTrans/rcdetr_90e_256×704_swinT.py 8 --work-dir work_dirs/xxx/
```
Evaluation
```
bash tools/dist_test.sh projects/configs/RCTrans/rcdetr_90e_256×704_swinT.py ckpts/xxx.pth 8 --eval bbox
```
Tracking
```
# following the scripts of CenterPoint.
```
Speed
```
python tools/benchmark.py projects/configs/test_speed/rcdetr_90e_256×704.py --checkpoint ckpts/xxx.pth
```
Visualize
```
python tools/visualize.py
# We also recommand to use the Visualization codes from BEVFormer, which is really nice.
```
## Weights
Download these backbones: [Swin_T](https://drive.usercontent.google.com/download?id=1OQhC-F4npQ4Dj9QIFUmWGE5Y56juLiEr&export=download&authuser=0&confirm=t&uuid=6b56dfd1-df54-4506-a9bc-1e088a76dfdf&at=APvzH3rsxTcnyR6_RLssyfXfLvhJ:1734079553818), [ResNet-18](https://drive.usercontent.google.com/download?id=1QWb74xrZ-HbywXvrLrYjs7hhCBheTS7n&export=download&authuser=0&confirm=t&uuid=6fb7c908-a33a-4bad-879f-25186fb67f14&at=APvzH3pcvUeKZrjbQ7WM818Dv41p:1734079499612), [ResNet-50](https://drive.usercontent.google.com/download?id=1LUg4Hjzn8BoOfjUTukHhsYj9Kj58PjE6&export=download&authuser=0&confirm=t&uuid=ea2707c9-dc11-4039-8436-18b4ee1c10ed&at=APvzH3r3SO-ITXZSXYCS8e8Tdc0y:1734079354810), [VovNet](https://drive.usercontent.google.com/download?id=17HVdkxE2nylUIU_mQrtexdG9nN8Mw2BN&export=download&authuser=0&confirm=t&uuid=19463c36-c860-4660-8f66-3c3fa60341bb&at=APvzH3oaxLom-XTmv-QfpCLOTx1O:1734079442549), and put them into the RCTrans/ckpts/.

We give the pre-trained in Table 1: [Swint-train](https://drive.usercontent.google.com/download?id=1SQZJ28rF7zs6-ARyvOWEuE1611WwvC0H&export=download&authuser=0&confirm=t&uuid=a69faea7-e47c-4176-b939-f0f36a628f15&at=APvzH3rOt8xJ4G33EnTHitPm7lal:1734080610522), [ResNet18-train](https://drive.usercontent.google.com/download?id=1zcvGfBU7j6eLi00ho0VXFCrZG1i5YEmL&export=download&authuser=0&confirm=t&uuid=8aced929-3827-4e9d-9968-ff7873b326a2&at=APvzH3pennsYULNE1cWgSQ1gMCaA:1734080088576), [ResNet50-train](https://drive.usercontent.google.com/download?id=17T3jGnjQhihL8dyptD7aNqSAfYhSl7YD&export=download&authuser=0&confirm=t&uuid=1015b59a-1f46-447b-80e5-504941f7aa1d&at=APvzH3rtLi-JFGxVie1YLA5_SEN6:1734080303675).

Details of test subset in Table 2 is [here](https://drive.usercontent.google.com/download?id=1A2QHEyIyMiSl9-swYOowr135i49trd6Y&export=download&authuser=0&confirm=t&uuid=ef523779-52d3-4088-9251-30fc39d7d915&at=AEz70l6JL9VtYWcuX9XRuSA-sBWe:1741957283379).
## Acknowledgements
We thank these great works and open-source codebases:
[MMDetection3d](https://github.com/open-mmlab/mmdetection3d), [BEVFormer](https://github.com/fundamentalvision/BEVFormer),
[DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR),
[StreamPETR](https://github.com/exiawsh/StreamPETR), [CMT](https://github.com/junjie18/CMT), [CenterPoint](https://github.com/tianweiy/CenterPoint), [FUTR3D](https://github.com/Tsinghua-MARS-Lab/futr3d).
## Citation
If you find our work is useful, please give this repo a star and cite our work as:
```bibtex
@article{li2024rctrans,
  title={RCTrans: Radar-Camera Transformer via Radar Densifier and Sequential Decoder for 3D Object Detection},
  author={Li, Yiheng and Yang, Yang and Lei, Zhen},
  journal={arXiv preprint arXiv:2412.12799},
  year={2024}
}
```