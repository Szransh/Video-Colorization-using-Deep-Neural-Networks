import os
import numpy as np 
import pandas as pd 
from PIL import Image
from torch.utils.data import Dataset
from skimage.color import rgb2lab, rgb2gray


class ImageDataset(Dataset):
    def __init__(self, root, captions_file, color_transform = None, transform = None):
        self.df = pd.read_csv(captions_file, index_col=None)
        self.transform = transform
        self.color_transform = color_transform
        self.images = self.df["image"]        
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.images[index])).convert("RGB")
        if self.color_transform:
          img = self.color_transform(img)
        img = np.array(img)
        
        img_lab = rgb2lab(img).astype("float32")
        lab_scaled =  (img_lab + 128) / 255
        

        L = rgb2gray(img)
        ab = lab_scaled[: , : , 1:]
        # print(L.shape)
        if self.transform:
            L = self.transform(L)
            ab = self.transform(ab)

        return L , ab