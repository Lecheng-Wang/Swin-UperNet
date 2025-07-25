# Swin-UperNet: Transformer-based Semantic Segmentation  

![Segmentation Example](Demo.png)  
*Example semantic segmentation results using Swin-UperNet*  

## Introduction  
Swin-UperNet is a state-of-the-art semantic segmentation framework that combines the powerful Swin Transformer backbone with the UperNet head architecture. This repository provides an implementation of the Swin-UperNet model for high-performance semantic segmentation tasks.  

**Key features**:  
- 🚀 Swin Transformer backbone for hierarchical feature extraction  
- 🔄 UperNet decoder for multi-scale feature fusion  
- ⚡️ High efficiency with linear computational complexity  
- 🏆 State-of-the-art performance on segmentation benchmarks  
- ❄️ Specialized support for glacier segmentation tasks  

## Installation  
### Prerequisites  
- Python 3.12.6  
- PyTorch 2.4.1
- TorchSummary 1.5.1
- TorchInfo 1.8.0
- Thop 0.1.1
- CUDA 11.8
- Timm 1.0.15
- numpy 1.26.4
- GDAL 3.8.4
- Linux environment recommended  

### Training Command
To train a model on the glacier segmentation task, use the following command structure:
```bash
python train.py \
    --MODEL_TYPE upernet \
    --BACKBONE_TYPE swin_t \
    --BANDS 10 \
    --NUM_CLASS 3 \
    --DATASET_PATH ./datasets/glacier \
    --BATCH_SIZE 16 \
    --EPOCHS 100 \
    --OPTIMIZER_TYPE sgd \
    --LOSS_TYPE ce \
    --LR_SCHEDULER poly \
    --INIT_LR 0.0005 \
    --GPU_ID 0
```
**Key Parameters**:
- `MODEL_TYPE`: Model architecture (e.g., `upernet`, `deeplab`, `segnext`, etc.)
- `BACKBONE_TYPE`: Backbone network (for models that support backbones, e.g., `swin_t`, `resnet50`, etc.)
- `BANDS`: Number of input channels (10 for glacier data)
- `NUM_CLASS`: Number of classes (including background)
- `DATASET_PATH`: Path to the dataset directory
- `BATCH_SIZE`: Batch size (adjust based on GPU memory)
- `EPOCHS`: Total training epochs
- `OPTIMIZER_TYPE`: Optimizer (`sgd` or `adam`)
- `LOSS_TYPE`: Loss function (`ce` for cross-entropy or `focal` for focal loss)
- `LR_SCHEDULER`: Learning rate scheduler (`poly`, `step`, `cos`, or `exp`)
- `INIT_LR`: Initial learning rate
- `GPU_ID`: ID of the GPU to use
**Note**: The dataset should be organized in the following structure:
```
DATASET_PATH/
├── annotations/
│   ├── train.txt
│   └── val.txt
├── images/
│   ├── 1.tif
│   ├── 2.tif
│   └── ...
└── labels/
    ├── 1.tif
    ├── 2.tif
    └── ...
```

## Comparative Models
All models were trained from scratch under identical conditions for fair comparison:

- `SETR` [19]
- `DeeplabV3+` [7]
- `HRNet` [20]
- `SegNeXt` [21]
- `Swin-UperNet` (ours)


