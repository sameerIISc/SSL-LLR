import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class Build_GPyr(nn.Module):

    def __init__(self, device='0'):
        super(Build_GPyr, self).__init__()
   
        self.cuda1 = torch.device('cuda:' + device)
        
        self.kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]], device=self.cuda1), requires_grad=False)
        self.kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]], device=self.cuda1), requires_grad=False)
        
        self.kernel = self.kernelv*self.kernelh*4
        self.kernel1 = self.kernelv*self.kernelh
        
    def forward(self, im, levels):
        
        gpyr = []
        gpyr.append(im)
        for _ in range(levels-1):
            gpyr.append(self.pyrReduce(gpyr[-1]))
        return  gpyr
        
    def pyrReduce(self, im):
        
        im_out = torch.zeros(im.size(0),3,int(im.size(2)/2),int(im.size(3)/2), device=self.cuda1)
       
        for k in range(3):
            
            temp = im[:,k,:,:].unsqueeze(dim=1)
            
            im_cp = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
            im_cp = torch.cat((im_cp, im_cp[:,:,:,-1].unsqueeze(dim=3), im_cp[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
            
            im_bp = torch.cat((im_cp[:,:,0,:].unsqueeze(dim=2), im_cp[:,:,0,:].unsqueeze(dim=2), im_cp), dim=2) # padding columns
            im_bp = torch.cat((im_bp, im_bp[:,:,-1,:].unsqueeze(dim=2), im_bp[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
            
            im1 = F.conv2d(im_bp, self.kernel1, padding = [0,0], groups=1)
            im_out[:,k,:,:] = im1[:,:,0::2,0::2]
        
        return im_out                 