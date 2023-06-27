import torch
import numpy as np
import numpy
from PIL import Image
import matplotlib as mp
from torch.autograd import Variable
import torch

def read_npy_as_tensor(path, device='0'):

    cuda1 = torch.device('cuda:' + device)
    
    sub = np.load(path, allow_pickle=False)
    sub = np.array(sub, dtype=np.float32)
    
    sub = numpy.moveaxis(sub, (0,1,2), (1,2,0))
    sub = torch.tensor(sub, device=cuda1, dtype=torch.float32)
    sub = sub.unsqueeze(dim=0)
    sub = sub[:,0:3,:,:]    

    return sub

def read_img_as_tensor(path, device='0'):

    cuda1 = torch.device('cuda:' + device)

    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    img = img/255        
    
    img = numpy.moveaxis(img, (0,1,2), (1,2,0))
    img = torch.tensor(img, device=cuda1, dtype=torch.float32)
    img = img.unsqueeze(dim=0)
    img = img[:,0:3,:,:]    

    return img

def read_img_as_npy(path):
    
    img = Image.open(path)
    img = np.array(img, dtype=np.float32)
    img = img/255        
    
    return img

def save_tensor_as_img(img, path):
    
    img = Variable(img, requires_grad=False).cpu().numpy()
    img = img[0,:,:,:]
    img = numpy.moveaxis(img, (0,1,2), (2,0,1))
    
    img[img>1] = 1
    img[img<0] = 0
    
    mp.image.imsave(path, img)
    
def save_npy_as_img(img, path):
    
    img[img>1] = 1
    img[img<0] = 0
    
    mp.image.imsave(path, img)
    
def save_2d_tensor_as_npy(sub, path):
        
    sub = Variable(sub, requires_grad=False).cpu().numpy()
    np.save(path, sub, allow_pickle=False)
    
def save_tensor_as_npy(sub, path):
    
    sub = Variable(sub, requires_grad=False).cpu().numpy()
    sub = sub[0,:,:,:]
    sub = numpy.moveaxis(sub, (0,1,2), (2,0,1))
    
    np.save(path, sub, allow_pickle=False)