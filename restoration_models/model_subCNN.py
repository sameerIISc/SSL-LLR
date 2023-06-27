import torch.nn as nn
import torch.nn.init as init

class SubCNN(nn.Module):
    def __init__(self, depth=10, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(SubCNN, self).__init__()
        
        kernel_size = 3
        padding = 1
        n_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu12 = nn.ReLU(inplace=True)
        
        self.conv13 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu14 = nn.ReLU(inplace=True)        
        
        self.conv15 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu15 = nn.ReLU(inplace=True)
        
        self.conv16 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu16 = nn.ReLU(inplace=True)                
        
        self.conv17 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu17 = nn.ReLU(inplace=True)        
        
        self.conv18 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.relu18 = nn.ReLU(inplace=True)        
        
        self.conv19 = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=kernel_size, padding=padding, bias=True)
        
        self._initialize_weights()

    def forward(self, x):

        # residual=x
        
        y1 = self.relu1(self.conv1(x))
        y2 = self.relu2(self.conv2(y1))
        y3 = self.relu3(self.conv3(y2))
        
        y3 = y3 + y1
        
        y4 = self.relu4(self.conv4(y3))
        y5 = self.relu5(self.conv5(y4))
        y6 = self.relu6(self.conv6(y5))
        
        y6 = y6 + y4
        
        y7 = self.relu7(self.conv7(y6))
        y8 = self.relu8(self.conv8(y7))
        y9 = self.relu9(self.conv9(y8))
        
        y9 = y9 + y7
        
        y10 = self.relu10(self.conv10(y9))
        y11 = self.relu11(self.conv11(y10))
        y12 = self.relu12(self.conv12(y11))
        
        y12 = y12 + y10
        
        y13 = self.relu13(self.conv13(y12))
        y14 = self.relu14(self.conv14(y13))
        y15 = self.relu15(self.conv15(y14))
        
        y15 = y15 + y13
        
        y16 = self.relu16(self.conv16(y15))
        y17 = self.relu17(self.conv17(y16))
        y18 = self.relu18(self.conv18(y17))
        
        y18 = y18 + y16
        
        out = self.conv19(y18)
                                
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