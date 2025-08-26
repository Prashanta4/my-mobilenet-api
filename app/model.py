import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class TripletAttention(nn.Module):
        def __init__(self, in_channels, kernel_size=7):
            super(TripletAttention, self).__init__()
            self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x_perm1 = x
            x_perm2 = x.permute(0, 2, 1, 3)
            x_perm3 = x.permute(0, 3, 2, 1)
            out1 = self._attention(x_perm1)
            out2 = self._attention(x_perm2).permute(0, 2, 1, 3)
            out3 = self._attention(x_perm3).permute(0, 3, 2, 1)
            out = (out1 + out2 + out3) / 3
            return out
        def _attention(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            pool = torch.cat([avg_out, max_out], dim=1)
            attn = self.conv1(pool)
            attn = self.sigmoid(attn)
            return x * attn
    
class SEBlock(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super(SEBlock, self).__init__()
            self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            w = nn.functional.adaptive_avg_pool2d(x, 1)
            w = self.relu(self.fc1(w))
            w = self.sigmoid(self.fc2(w))
            return x * w
    
class ECABlock(nn.Module):
        def __init__(self, channels, k_size=3):
            super(ECABlock, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            y = self.avg_pool(x)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            return x * y.expand_as(x)
    
class RESBlock(nn.Module):
        def __init__(self, in_channels):
            super(RESBlock, self).__init__()
            self.se = SEBlock(in_channels)
            self.eca = ECABlock(in_channels)
        def forward(self, x):
            out_se = self.se(x)
            out_eca = self.eca(x)
            return out_se + out_eca
    
class ModifiedMobileNetV2(nn.Module):
        def __init__(self, num_classes=10, insert_indices=(3, 5, 8, 10, 13, 15)):
            super().__init__()
            base = mobilenet_v2(weights='DEFAULT')
            self.features = nn.Sequential()
            attention_count = 0
            resblock_count = 0
            ta_insert_points = set([3, 8, 13])
            res_insert_points = set([5, 10, 15])
            for idx, layer in enumerate(base.features):
                self.features.add_module(str(idx), layer)
                out_channels = None
                if hasattr(layer, 'out_channels'):
                    out_channels = layer.out_channels
                elif hasattr(layer, 'conv'):
                    out_channels = layer.conv[-1].out_channels
                else:
                    out_channels = layer[0].out_channels
                if idx in ta_insert_points:
                    self.features.add_module(f'ta{attention_count+1}', TripletAttention(out_channels))
                    attention_count += 1
                if idx in res_insert_points:
                    self.features.add_module(f'res{resblock_count+1}', RESBlock(out_channels))
                    resblock_count += 1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(base.last_channel, num_classes)
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x