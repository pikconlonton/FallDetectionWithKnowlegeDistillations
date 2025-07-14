# @title IMPORT_LIBRARY
import  torch
import  matplotlib.pyplot as plt
import numpy as np
from PIL import  Image
import os
from torchvision.transforms  import Compose , Resize,ToTensor,ToPILImage
from torch.utils.data import  Dataset,DataLoader
import cv2
from PIL import Image

# @title MAKE DATASET FOR CLASSIFICATION
class Falling_Dataset4Clss(Dataset):
  def __init__(self, root, type_data, transform=None):
    self.transform = transform
    self.root = root + '/' + type_data + '/'
    self.img_paths = []
    self.labels = []
    self.labels_name = ['fall', 'normal']
    with open(self.root + 'labels.txt', 'r') as file:
      list_labels = file.read()

    self.labels = [int(x) for x in list_labels.split() if x.isdigit()]

    self.img_paths = [self.root + 'images/' + f for f in os.listdir(self.root + '/images')]
    self.img_paths = sorted(self.img_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))


  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    image = Image.open(self.img_paths[index]).convert('RGB')
    if self.transform:
      image = self.transform(image)

    # path_splited = list(self.img_paths[index].split('/'))
    # name_img = path_splited[-1].split('.')[0]
    # try:
    #     label = self.labels[int(name_img)]
    # except IndexError:
    #     print(f"Lỗi: name_img = {name_img}, int(name_img) = {int(name_img)}, len(labels) = {len(self.labels)}")
    #     raise
    label = self.labels[int(index)]
    return image, torch.tensor(label)







