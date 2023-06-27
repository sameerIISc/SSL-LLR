import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class reconLPyr(nn.Module):
    def __init__(self, device='0'):
        super(reconLPyr, self).__init__()
    
        self.cuda1 = torch.device('cuda:' + device)
        
        self.kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]], device=self.cuda1), requires_grad=False)
        self.kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]], device=self.cuda1), requires_grad=False)
        
        self.kernel = self.kernelv*self.kernelh*4
        self.kernel1 = self.kernelv*self.kernelh
        
        self.ker00 = self.kernel[:,:,0::2,0::2]
        self.ker01 = self.kernel[:,:,0::2,1::2]
        self.ker10 = self.kernel[:,:,1::2,0::2]
        self.ker11 = self.kernel[:,:,1::2,1::2]
    
    def forward(self, subs):
        
        img = subs.pop()
        while len(subs)>0:
            img = subs.pop() + self.pyrExpand(img) 
        return img
    
    def pyrExpand(self, im):
        
        out = torch.zeros(im.size(0),im.size(1),im.size(2)*2,im.size(3)*2, device=self.cuda1, dtype=torch.float32)
        
        for k in range(3):
            
            temp = im[:,k,:,:]
            temp = temp.unsqueeze(dim=1)
                           
            im_c1 = torch.cat((temp, temp[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
            im_c1r1 = torch.cat((im_c1, im_c1[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                    
            im_r2 = torch.cat((temp[:,:,0,:].unsqueeze(dim=2), temp), dim=2) # padding columns
            im_r2 = torch.cat((im_r2, im_r2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
            
            im_r2c1 = torch.cat((im_r2, im_r2[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
                    
            im_c2 = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
            im_c2 = torch.cat((im_c2, im_c2[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
            
            im_c2r1 = torch.cat((im_c2, im_c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                    
            im_r2c2 = torch.cat((im_c2[:,:,0,:].unsqueeze(dim=2), im_c2), dim=2) # padding columns
            im_r2c2 = torch.cat((im_r2c2, im_r2c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
                    
            im_00 = F.conv2d(im_r2c2, self.ker00, padding = [0,0], groups=1)
            im_01 = F.conv2d(im_r2c1, self.ker01, padding = [0,0], groups=1)
            im_10 = F.conv2d(im_c2r1, self.ker10, padding = [0,0], groups=1)
            im_11 = F.conv2d(im_c1r1, self.ker11, padding = [0,0], groups=1)
            
            out[:,k,0::2,0::2] = im_00
            out[:,k,1::2,0::2] = im_10
            out[:,k,0::2,1::2] = im_01
            out[:,k,1::2,1::2] = im_11
                     
        return out             