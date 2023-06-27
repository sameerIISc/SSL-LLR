import torch
import torch.nn as nn
from torchvision.models import resnet18

class module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(module, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2, bias=True)
        self.dp1 = nn.Dropout2d(p=0.1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.dp2 = nn.Dropout2d(p=0.1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        out = self.relu1(self.dp1(self.bn1(self.conv1(x))))
        out = self.relu2(self.dp2(self.bn2(self.conv2(out))))
        
        return out
    
class qfcnn(nn.Module):
    def __init__(self):
        super(qfcnn, self).__init__()
        
        self.features = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=True)
        
        self.model = nn.Sequential(
            module(32, 64),
            module(64, 128),
            module(128, 128))
        
        self.apool = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self,x):
        
        features = self.features(x)
        out = self.model(features)
        
        out = self.apool(out)
        
        out = torch.flatten(out, start_dim=1)
        
        return out
