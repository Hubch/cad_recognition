import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights, MobileNet_V2_Weights

class HybridConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.3, init=True, update=False, weight_path=None, cuda=True):
        super(HybridConvNeXtTiny, self).__init__()
        
        self.init = init
        self.update = update
        self.weight_path = weight_path
        self.cuda = cuda
        
        # Load the pre-trained ConvNeXt_Tiny
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # Modify the first convolution layer to accept 1-channel input
        self.model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0, bias=False)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feature = nn.Linear(768, 512)
        self.fc_classification = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        
        if not self.init and self.weight_path:
            self.load_weight()
            
        if self.update:
            self.freeze_layers()
        
    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x).view(x.size(0), -1)
        feature = self.dropout(self.fc_feature(x))
        logits = self.dropout(self.fc_classification(x))
        
        return feature, logits
    
    def load_weight(self):
        if self.cuda:
            self.load_state_dict(torch.load(self.weight_path + "_teacher_best.pth"))
        else:
            self.load_state_dict(torch.load(self.weight_path + "_teacher_best.pth", map_location=torch.device('cpu')))
        print(f'load weight from {self.weight_path}_teacher_best.pth')
    
    def freeze_layers(self):
        for name, param in self.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

                
class HybridMobileNetV2(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.3, init=True, update=False, weight_path=None, cuda=True):
        super(HybridMobileNetV2, self).__init__()
        
        self.init = init
        self.update = update
        self.weight_path = weight_path
        self.cuda = cuda
        
        # Load the pre-trained MobileNetV2
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Modify the first convolution layer to accept 1-channel input
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify the classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feature = nn.Linear(1280, 512)
        self.fc_classification = nn.Linear(1280, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        
        if not self.init and self.weight_path:
            self.load_weight()
            
        if self.update:
            self.freeze_layers()
        
    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x).view(x.size(0), -1)
        feature = self.dropout(self.fc_feature(x))
        logits = self.dropout(self.fc_classification(x))
        
        return feature, logits
    
    def load_weight(self):
        if self.cuda:
            self.load_state_dict(torch.load(self.weight_path + "_student_best.pth"))
        else:
            self.load_state_dict(torch.load(self.weight_path + "_student_best.pth", map_location=torch.device('cpu')))
        print(f'load weight from {self.weight_path}_student_best.pth')
    
    def freeze_layers(self):
        for name, param in self.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
                
def get_teacher_model(num_classes=0, dropout_p=0.3, init=True, update=False, weight_path=None, cuda=True):
    return HybridConvNeXtTiny(num_classes, dropout_p, init, update, weight_path, cuda)

def get_student_model(num_classes=0, dropout_p=0.3, init=True, update=False, weight_path=None, cuda=True):
    return HybridMobileNetV2(num_classes, dropout_p, init, update, weight_path, cuda)
