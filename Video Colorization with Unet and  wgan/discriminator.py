import numpy as np  
import torch
from torch import nn, optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels =64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=True),
            
            nn.Conv2d(out_channels, out_channels*2**1, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels*2**1, affine=True),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(out_channels*2**1, out_channels*2**2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels*2**2, affine=True),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(out_channels*2**2, out_channels*2**3, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels*2**3, affine=True),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(out_channels*2**3, 1, 4, stride=1, padding=1, bias=True),
        ) 
    
    def forward(self, x):
        return self.layers(x)