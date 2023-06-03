import numpy as np  
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import numpy as np  
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

#Load model
from google.colab import drive
import shutil
def save_model(model):
  #Saving the best model to drive
  drive.mount('/content/drive')
  shutil.copy("/content/" + model, "/content/drive/MyDrive/cv thesis/wgan/model")
  print("Model Saved")
  drive.flush_and_unmount()

def load_model(model):
  drive.mount('/content/drive')
  shutil.copy("/content/drive/MyDrive/cv thesis/wgan/model/" + model, '/content/')
  print("Model Loaded")
  drive.flush_and_unmount()
  
  