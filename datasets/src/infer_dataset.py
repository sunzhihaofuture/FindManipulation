import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class InferDataset(Dataset):
    def __init__(self, dataset_rootpath, labelfile_paths, image_size=512):
        self.dataset_rootpath = dataset_rootpath
        self.labelfile_paths = labelfile_paths
        self.infer_transforms = A.Compose([
            A.Resize(height=image_size, width=image_size, p=1.0),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
        self.image_path_list, self.mask_path_list = [], []

        for labelfile_path in self.labelfile_paths:
            labelfile = open(labelfile_path, 'r')
            for line in labelfile:
                infos = line[:-1].split(' ')

                image_path = f'{dataset_rootpath}/{infos[0]}'
                mask_path = f'{dataset_rootpath}/{infos[1]}'

                if os.path.exists(image_path) and (('none' in mask_path) or os.path.exists(mask_path)):
                    self.image_path_list.append(image_path)
                    self.mask_path_list.append(mask_path)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        mask_path = self.mask_path_list[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image / 255.

        if 'none' in mask_path:
            mask = np.zeros((image.shape[0], image.shape[1]))
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.

        augments = self.infer_transforms(image=image, mask=mask)
        image, mask = augments['image'].float(),  augments['mask'][None, ...].float()
        
        return image, mask, image_path, mask_path

    def __len__(self):
        return len(self.image_path_list)
