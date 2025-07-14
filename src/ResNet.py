

# @title IMPORT_LIBRARY
import torch
from torch.utils.data import TensorDataset # Import TensorDataset
from torchvision.datasets  import CIFAR10
import torch.nn as nn
from torchsummary import  summary
from torchvision.transforms  import Compose , Resize,ToTensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from tqdm import  tqdm  #Tạo thanh Progress Bar
from sklearn.metrics import accuracy_score


# @title MODEL RESNET 18
class BasicBlock(nn.Module):
  def __init__(self, in_channels,out_channels, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=False)
    self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1,padding=1,bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride,bias = False),
          nn.BatchNorm2d(out_channels)

      )
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)  # ReLU should not modify the tensor in-place
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)  # Again, no in-place operation

    out = out + self.shortcut(x)
    out = self.relu(out)
    return out


class Resnet18(nn.Module):
  def __init__(self, num_classes = 10):
    super(Resnet18,self).__init__()
    self.in_channels = 64
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1,padding=1, bias=False) #Giu nguyen size vi Cfar size be
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=False)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self.make_layer(64,BasicBlock, 2,   stride=1)
    self.layer2 = self.make_layer(128,BasicBlock, 2,  stride=2)
    self.layer3 = self.make_layer(256,BasicBlock, 2,  stride=2)
    self.layer4 = self.make_layer(512,BasicBlock, 2,  stride=2)

    self.avg = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512, num_classes)


  def make_layer(self, out_channels,block, num_block, stride):
    strides = [stride] + [1]*(num_block-1)
    layer = []
    for stride in strides:
      layer.append(block(self.in_channels,out_channels, stride))
      self.in_channels = out_channels

    return nn.Sequential(*layer)

  def forward(self,x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.maxpool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avg(out)
    out = out.view(out.size(0),-1)
    out = self.fc(out)
    return out

