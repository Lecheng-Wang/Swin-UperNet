# Swin Transformer-Based UperNet for Glacier Semantic Segmentation

![Segmentation Example](Swin_UperNet.png)  
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

## Comparative Models links
All models were trained from scratch under identical conditions for fair comparison:

- **`DeepLabv3+`**  
  **Encoder-Decoder with Atrous Separable Convolution (Liang-Chieh Chen et al., 2018, ECCV).**  
  Paper: [arXiv:1802.02611 (PDF)](https://arxiv.org/pdf/1802.02611).  
- **`U-Net`**  
  **U-Net: Convolutional Networks for Biomedical Image Segmentation (Olaf Ronneberger et al., 2015, MICCAI).**  
  Paper: [arXiv:1505.04597 (PDF)](https://arxiv.org/pdf/1505.04597).  
- **`SegFormer`**  
  **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (Enze Xie et al., 2021, NIPS).**  
  Paper: [arXiv:2105.15203 (PDF)](https://arxiv.org/pdf/2105.15203).  
- **`SETR`**  
  **Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers (Sixiao Zheng et al., 2021, CVPR).**  
  Paper: [arXiv:2012.15840 (PDF)](https://arxiv.org/pdf/2012.15840).  
- **`Swin-UperNet`** (ours)  
  **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Ze Liu et al., 2021, ICCV).**  
  Paper: [arXiv:2103.14030 (PDF)](https://arxiv.org/pdf/2103.14030).  
  **Unified Perceptual Parsing for Scene Understanding (UPerNet) (Tete Xiao et al., 2018, ECCV).**   
  Paper: [arXiv:1807.10221 (PDF)](https://arxiv.org/pdf/1807.10221).  



