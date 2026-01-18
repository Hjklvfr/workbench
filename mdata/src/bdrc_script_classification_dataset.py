import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class BDRCScriptClassification(Dataset):
    def __init__(self, ds_path, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []

        classes = list(filter(lambda x: '.' not in x, [f for f in os.listdir(ds_path) if os.path.isdir(ds_path)]))
        
        self.labels_to_idx = {
            label: i for i, label in enumerate(sorted(set(classes)))
        }

        for class_ in classes:
            path = f"{ds_path}/{class_}"
            samples_paths = list(filter(lambda x: '.jpg' in x.lower(), [f"{path}/{f}" for f in os.listdir(path) if os.path.isdir(path)]))
            self.paths += samples_paths
            self.labels += [class_] * len(samples_paths)
    
    def __len__(self):
        # return len(self.paths)
        return 3
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert('1')
        if self.transform:
            image = self.transform(image)
            # print(f"image shape: {image.shape}")
        label = self.labels[idx]
        return image, torch.tensor(self.labels_to_idx[label], dtype=torch.long)