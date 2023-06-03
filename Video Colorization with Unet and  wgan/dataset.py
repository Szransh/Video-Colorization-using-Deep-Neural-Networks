from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import numpy as np
class DATASET(Dataset):
    def __init__(self, paths, transform = None):
        self.transform = transform            
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        L = img_lab[:,:,0] / 50. - 1.
        ab = img_lab[:,:,1:] / 128.
        L = transforms.ToTensor()(L)
        ab = transforms.ToTensor()(ab)  
                     
        return L, ab
    
    def __len__(self):
        return len(self.paths)

