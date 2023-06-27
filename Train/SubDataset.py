import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
    
class SubDataset(Dataset):
    def __init__(self, short_dir, long_dir, dataset, ps, device='0'):
       
        self.files_short = os.listdir(short_dir)
        self.cuda1 = torch.device('cuda:' + device)
        self.short_dir = short_dir
        self.long_dir = long_dir
        self.dataset = dataset
        self.ps = ps
        
    def __len__(self):
        return len(self.files_short)

    def read_data(self, path):
        
        sub = np.load(path, allow_pickle=False)
        sub = np.moveaxis(sub, (0,1,2), (1,2,0))

        return sub

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
        
        if self.dataset == 'lol':
            path_long = self.long_dir + name_short
            sub_long = self.read_data(path_long)
        else:
            path_long = glob.glob(self.long_dir + name_short[:5] + '*.npy')
            sub_long = self.read_data(path_long[0])
        
        short_patch, long_patch = self.extract_patches(sub_short, sub_long)
        
        return short_patch, long_patch
    