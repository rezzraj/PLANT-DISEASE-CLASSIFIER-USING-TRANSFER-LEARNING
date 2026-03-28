import os
import shutil
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms , models
from torchvision.datasets import ImageFolder
from torch import nn
import optuna
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision.models import MobileNet_V2_Weights
import torch.optim as optim
import multiprocessing


class MobileNetV2(nn.Module):
    def __init__(self, neurons_per_hidden_layer, dropout, num_classes):
        super().__init__()
        # loading pretrained mobileNetV2
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        image_channel = self.model.last_channel
        layer = []

        for neurons in neurons_per_hidden_layer:
            layer.append(nn.Linear(image_channel, neurons))
            layer.append(nn.ReLU())
            layer.append(nn.BatchNorm1d(neurons))
            layer.append(nn.Dropout(dropout))
            image_channel = neurons
        layer.append(nn.Linear(image_channel, num_classes))
        self.model.classifier = nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x)