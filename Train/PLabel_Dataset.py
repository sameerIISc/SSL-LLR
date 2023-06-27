import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

## creating the dataset
class PLabel_Dataset(Dataset):
    def __init__(self, dataset, short_dir, long_dir, ps, device='0'):

        self.long_dir = long_dir
        self.short_dir = short_dir        
        self.dataset = dataset
        self.ps = ps
        self.cuda1 = torch.device('cuda:' + device)

        if dataset == 'lol':
            self.files_short = os.listdir(long_dir)
        else:
            self.files_short = os.listdir(short_dir)
        
    def __len__(self):
        return len(self.files_short)

    def read_data(self, path):
        img = cv2.imread(path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, (0,1,2), (1,2,0))
        img = img.astype(np.float32)
        sub = img/255
        return sub
    
    def compute_exposure(self, name_short):

        exp = ''
        for count, letter in enumerate(name_short):
            if letter == 's':
                exp = list(name_short)[count-1]
                
                if exp == '3':
                    exposure = '0.033'
                elif exp == '4':
                    exposure = '0.04'
                else:
                    exposure = '0.1'                 
                break
            
        return exposure
                    
    def extract_patches(self, sub_short, sub_long):

        H = sub_short.shape[1]
        W = sub_short.shape[2]
        
        xx = np.random.randint(0,W-self.ps)
        yy = np.random.randint(0,H-self.ps)
        short_patch = sub_short[:,yy:yy+self.ps,xx:xx+self.ps]
        long_patch = sub_long[:,yy:yy+self.ps,xx:xx+self.ps]
        
        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            short_patch = np.flip(short_patch, axis=1)
            long_patch = np.flip(long_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            short_patch = np.flip(short_patch, axis=0)
            long_patch = np.flip(long_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            short_patch = np.transpose(short_patch, (0,2,1))
            long_patch = np.transpose(long_patch, (0,2,1))
            
        short_patch = torch.tensor(short_patch.copy(), device=self.cuda1, dtype=torch.float32)
        long_patch = torch.tensor(long_patch.copy(), device=self.cuda1, dtype=torch.float32)
        
        return short_patch, long_patch
        
    def __getitem__(self, idx):
       
        name_short = self.files_short[idx]
        path_short = self.short_dir + name_short
        sub_short = self.read_data(path_short)

        if not self.dataset == 'lol':
            img_id = name_short[:5]
            exposure = self.compute_exposure(name_short)
            path_long = self.long_dir + img_id + '_00_' + exposure + 's.png'
        else:
            path_long = self.long_dir + name_short
  
        sub_long = self.read_data(path_long)

        short_patch, long_patch = self.extract_patches(sub_short, sub_long)
        
        return short_patch[:3,:,:], long_patch[:3,:,:]