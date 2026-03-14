#模型训练（水域分割）
# --BACKBONE_TYPE swin_t
# --BACKBONE_TYPE resnet50
python train_water.py `
    --TRAIN_IMAGE_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\train\images" `
    --TRAIN_MASK_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\train\masks" `
    --VAL_IMAGE_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\val\images" `
    --VAL_MASK_DIR "D:\Files\GitProject\BiSeNet-ooooverflow-LY\dataset\water_seg2\val\masks" `
    --MODEL_TYPE upernet `
    --BACKBONE_TYPE swin_t `
    --BANDS 3 `
    --NUM_CLASS 2 `
    --IMG_SIZE 256 `
    --BATCH_SIZE 16 `
    --EPOCHS 10 `
    --OPTIMIZER_TYPE sgd `
    --LOSS_TYPE ce `
    --LR_SCHEDULER poly `
    --INIT_LR 0.0005 `
    --GPU_ID 0

