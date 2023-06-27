import torch.nn as nn
import torch.nn.init as init
import torch 

cuda1 = torch.device('cuda:0')

class instance_norm(nn.Module):
    def __init__(self):
        super(instance_norm, self).__init__()

        self.scaler = nn.Parameter(torch.randn((1,64,1,1), device=cuda1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((1,64,1,1), device=cuda1, dtype=torch.float32))
        
    def forward(self, x, pert):

        std_ = torch.std(x, dim=(2,3), keepdim=True)
        mu_ = torch.mean(x, dim=(2,3), keepdim=True)
        
        x = (x - mu_)/std_

        x = x*(pert*self.scaler) + self.bias
        
        return x

class res_mod(nn.Module):
    def __init__(self):
        super(res_mod, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        
        res = x
        
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        
        out = out + res
        
        return out

class res_mod_ex(nn.Module):
    def __init__(self):
        super(res_mod_ex, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)


    def forward(self, x):
        
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out3 = self.relu3(self.conv3(out2))
        
        out = out3 + out1
        
        return out

class norm_mod(nn.Module):
    def __init__(self):
        super(norm_mod, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.norm = instance_norm()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, pert):
        
        out = self.relu(self.norm(self.conv(x), pert))
        
        return out
        
class lpCNN(nn.Module):
    def __init__(self):
        super(lpCNN, self).__init__()
        
        self.extract_features = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=True)

        self.norm1 = norm_mod()
        self.res1 = res_mod()
        
        self.norm2 = norm_mod()
        self.res2 = res_mod()

        self.norm3 = norm_mod()
        self.res3 = res_mod()

        self.res4 = res_mod_ex()
        self.res5 = res_mod_ex()
        self.res6 = res_mod_ex()

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True)
        
        self._initialize_weights()

    def forward(self, x):
    
        features = self.extract_features(x)
        
        y1 = self.res1(self.norm1(features, 1))
        y2 = self.res2(self.norm2(y1, 1))
        y3 = self.res3(self.norm3(y2, 1))
        
        y4 = self.res4(y3)
        y5 = self.res5(y4)
        y6 = self.res6(y5)
        
        out = self.conv_out(y6)
                        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)