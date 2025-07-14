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

# @title EfficientNet-B0 MODEL
import torch
import torch.nn as nn

# Swish Activation
class Swish(nn.Module):
    def forward(self, x):
        result = x * torch.sigmoid(x)
        if torch.isnan(result).any():
            raise ValueError("Gặp phải giá trị NaN trong kích hoạt Swish.")
        return result

# SE Block (Squeeze and Excitation)
class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SE_Block, self).__init__()
        hidden_dim = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (C, H, W) -> (C, 1, 1)
            nn.Conv2d(in_channels, hidden_dim, 1),
            Swish(),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# MBConv Block
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConv, self).__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.has_residual = (stride == 1 and in_channels == out_channels)
        self.has_se = se_ratio is not None and se_ratio > 0
        self.expand_ratio = expand_ratio

        # Expansion layer
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ) if expand_ratio != 1 else nn.Identity()

        # Depthwise convolution layer
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )

        # Squeeze-and-Excitation block
        if self.has_se:
            self.se = SE_Block(hidden_dim, reduction=int(1 // se_ratio))

        # Project layer
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        if self.has_se:
            out = self.se(out)
        out = self.project(out)
        if self.has_residual:
            out = out + x  # Skip connection
        return out


# EfficientNet-B0
class My_EfficientNet_B0(nn.Module):
    def __init__(self, num_classes):
        super(My_EfficientNet_B0, self).__init__()
        base_channels = 32
        self.steam = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            Swish()
        )

        in_channels = base_channels
        layers = []

        config = [
            (1, 16, 1, 3, 1),
            (6, 24, 2, 3, 2),
            (6, 40, 2, 5, 2),
            (6, 80, 3, 3, 2),
            (6, 112, 3, 5, 1),
            (6, 192, 4, 5, 2),
            (6, 320, 1, 3, 1)
        ]

        # Chỉnh sửa cấu hình để đảm bảo sự phù hợp về kênh
        for expand_ratio, out_channels, num_layers, kernel_size, stride in config:
            for i in range(num_layers):
                s = stride if i == 0 else 1
                layers.append(MBConv(in_channels, out_channels, kernel_size, s, expand_ratio))
                in_channels = out_channels

        self.body = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.steam(x)
        x = self.body(x)
        x = self.head(x)
        return x


