# encoding = utf-8

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class Labeled_Model_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, bands=3, img_size=256):
        super(Labeled_Model_Dataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bands = bands
        self.img_size = img_size  # 添加图像尺寸参数
        
        # 获取所有图像文件名
        self.image_names = [f for f in os.listdir(image_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'))]
        self.image_names.sort()
        self.length = len(self.image_names)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = self.image_names[index]
        name_wo_ext = os.path.splitext(img_name)[0]
        
        # 构建图像路径
        img_path = os.path.join(self.image_dir, img_name)
        
        # 标签可能有不同扩展名，尝试匹配
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        mask_path = None
        for ext in mask_extensions:
            potential_path = os.path.join(self.mask_dir, name_wo_ext + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for {name_wo_ext} in {self.mask_dir}")
        
        # 读取图像并调整尺寸
        image = Image.open(img_path)
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32)
        
        # 处理单波段或多波段
        if len(image.shape) == 2:
            image = np.stack([image] * self.bands, axis=0)
        else:
            image = image.transpose(2, 0, 1)
            if image.shape[0] < self.bands:
                repeat_times = self.bands // image.shape[0] + 1
                image = np.tile(image, (repeat_times, 1, 1))[:self.bands]
            elif image.shape[0] > self.bands:
                image = image[:self.bands]
        
        # 归一化到 0-1
        image = image / 255.0
        
        # 读取标签并调整尺寸（使用最近邻插值保持标签值）
        label = Image.open(mask_path)
        label = label.resize((self.img_size, self.img_size), Image.NEAREST)
        label = np.array(label)
        
        if len(label.shape) == 3:
            label = label[:, :, 0]
        
        # 标签映射：二分类（水/非水）
        label = (label > 128).astype(np.int64)
        
        return torch.from_numpy(image), torch.from_numpy(label).long()


class UnLabeled_Model_Dataset(Dataset):
    def __init__(self, image_dir, bands=3, img_size=256):
        super(UnLabeled_Model_Dataset, self).__init__()
        self.image_dir = image_dir
        self.bands = bands
        self.img_size = img_size
        
        self.image_names = [f for f in os.listdir(image_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'))]
        self.image_names.sort()
        self.length = len(self.image_names)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 读取图像并调整尺寸
        image = Image.open(img_path)
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32)
        
        # 处理单波段或多波段
        if len(image.shape) == 2:
            image = np.stack([image] * self.bands, axis=0)
        else:
            image = image.transpose(2, 0, 1)
            if image.shape[0] < self.bands:
                repeat_times = self.bands // image.shape[0] + 1
                image = np.tile(image, (repeat_times, 1, 1))[:self.bands]
            elif image.shape[0] > self.bands:
                image = image[:self.bands]
        
        image = image / 255.0
        
        return torch.from_numpy(image)