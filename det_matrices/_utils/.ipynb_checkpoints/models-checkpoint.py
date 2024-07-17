import torch
import torch.nn as nn
import torchvision.models as models


# 定义CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        return x

class MultiScaleResNeXt50(nn.Module):
    def __init__(self, init=True, update=False, weight_path=None):
        super(MultiScaleResNeXt50, self).__init__()
        self.init = init
        self.update = update
        self.weight_path = weight_path
        
        self.resnext50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if init else None)
        self.resnext50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)
        self.cbam5 = CBAM(2048)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048 + 1024 + 512 + 256 + 64, 1280)
        
        # Initialize weights if required
        if not self.init:
            self.weight_load(init=self.init, weight_path=self.weight_path)
        
        # Freeze feature extractor layers if update is False
        if update:
            self.freeze_extractor(update=self.update)

    def weight_load(self, init=True, weight_path=None):
        # Load weights from specified path
        if weight_path:
            self.load_state_dict(torch.load(weight_path))
        else:
            raise ValueError("weight_path must be provided when init is False")

    def freeze_extractor(self, update=False):
        for name, param in self.feature_extractor.named_parameters():
            if 'fc' not in name:  # Only freeze the feature extractor layers
                param.requires_grad = False
    
    def forward(self, x):
        x1 = self.resnext50.conv1(x)
        x1 = self.resnext50.bn1(x1)
        x1 = self.resnext50.relu(x1)
        x1 = self.resnext50.maxpool(x1)
        x1 = self.cbam1(x1)  # Apply CBAM

        x2 = self.resnext50.layer1(x1)
        x2 = self.cbam2(x2)  # Apply CBAM
        
        x3 = self.resnext50.layer2(x2)
        x3 = self.cbam3(x3)  # Apply CBAM
        
        x4 = self.resnext50.layer3(x3)
        x4 = self.cbam4(x4)  # Apply CBAM
        
        x5 = self.resnext50.layer4(x4)
        x5 = self.cbam5(x5)  # Apply CBAM

        x1 = self.pool(x1).view(x1.size(0), -1)
        x2 = self.pool(x2).view(x2.size(0), -1)
        x3 = self.pool(x3).view(x3.size(0), -1)
        x4 = self.pool(x4).view(x4.size(0), -1)
        x5 = self.pool(x5).view(x5.size(0), -1)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.fc(x)
        return x
    

def get_model(init, update, weight_path=None):
    return MultiScaleResNeXt50(init, update, weight_path)