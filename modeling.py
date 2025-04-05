import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
# 加载预训练的VGG19模型
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']  # 选择用于风格迁移的特征层
         
        # 获取完整的VGG19特征层
        full_model = vgg19(weights=VGG19_Weights.DEFAULT).features
         
        # 创建一个新的nn.Sequential对象，并添加前29层
        self.model = nn.Sequential(*list(full_model.children())[:29])
 
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features