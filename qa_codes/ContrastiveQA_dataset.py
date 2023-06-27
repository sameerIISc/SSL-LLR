import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import cv2

 ## creating the dataset
class ContrastiveQA_dataset(Dataset):
    def __init__(self, img_dir, prtrbtn_dir, dataset, device='0'):
        
        self.img_dir = img_dir
        self.prtrbtn_dir = prtrbtn_dir
        self.dataset = dataset
        
        self.files_img = os.listdir(img_dir)
        self.cuda1 = torch.device('cuda:' + device)

    def __len__(self):
        # return size of dataset
        return len(self.files_img)

    def read_data(self, path):
         img = cv2.imread(path)
         img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         img = np.moveaxis(img, (0,1,2), (1,2,0))
         img = img.astype(np.float32)
         sub = img/255
         return sub
 
    def __getitem__(self, idx):
       
        name_img = self.files_img[idx]
        prtrbd_paths = glob(self.prtrbtn_dir + name_img[:-4] + '_*.png')
        
        num_anchor = np.random.randint(len(prtrbd_paths))
        
        anchor_path = prtrbd_paths[num_anchor]
        im_anchor = self.read_data(anchor_path)
        
        H = im_anchor.shape[1]
        W = im_anchor.shape[2]
        
        ps_x = int(H/2)
        ps_y = int(W/2)
        
        if self.dataset == 'lol':
            ps_y = 37
        
        ######## reading views of anchor image ###########
        
        anchor_view1 = im_anchor[:3, :ps_x, :ps_y] 
        anchor_view2 = im_anchor[:3, ps_x:, :ps_y]
        
        if self.dataset == 'lol':
            anchor_view3 = im_anchor[:3, :ps_x, ps_y:-1] 
            anchor_view4 = im_anchor[:3, ps_x:, ps_y:-1] 
        else:
            anchor_view3 = im_anchor[:3, :ps_x, ps_y:] 
            anchor_view4 = im_anchor[:3, ps_x:, ps_y:] 
            
        ######## reading negatives ##########
        
        negatives_view1 = np.zeros((len(prtrbd_paths)-1,3,ps_x,ps_y), dtype=np.float32)
        negatives_view2 = np.zeros((len(prtrbd_paths)-1,3,ps_x,ps_y), dtype=np.float32)
        negatives_view3 = np.zeros((len(prtrbd_paths)-1,3,ps_x,ps_y), dtype=np.float32)
        negatives_view4 = np.zeros((len(prtrbd_paths)-1,3,ps_x,ps_y), dtype=np.float32)
        count = 0
        for path in prtrbd_paths:
            
            if not path == anchor_path:
                
                img = self.read_data(path)
        
                negatives_view1[count,:,:,:] = img[:3, :ps_x,:ps_y]
                negatives_view2[count,:,:,:] = img[:3, ps_x:, :ps_y]
                if self.dataset =='lol':
                    negatives_view3[count,:,:,:] = img[:3, :ps_x, ps_y:-1]
                    negatives_view4[count,:,:,:] = img[:3, ps_x:, ps_y:-1]
                else:
                    negatives_view3[count,:,:,:] = img[:3, :ps_x, ps_y:]
                    negatives_view4[count,:,:,:] = img[:3, ps_x:, ps_y:]
               
                count += 1
        
        anchor_view1 = torch.tensor(anchor_view1.copy(), device=self.cuda1, dtype=torch.float32)
        anchor_view2 = torch.tensor(anchor_view2.copy(), device=self.cuda1, dtype=torch.float32)
        anchor_view3 = torch.tensor(anchor_view3.copy(), device=self.cuda1, dtype=torch.float32)
        anchor_view4 = torch.tensor(anchor_view4.copy(), device=self.cuda1, dtype=torch.float32)
        negatives_view1 = torch.tensor(negatives_view1.copy(), device=self.cuda1, dtype=torch.float32)
        negatives_view2 = torch.tensor(negatives_view2.copy(), device=self.cuda1, dtype=torch.float32)
        negatives_view3 = torch.tensor(negatives_view3.copy(), device=self.cuda1, dtype=torch.float32)
        negatives_view4 = torch.tensor(negatives_view4.copy(), device=self.cuda1, dtype=torch.float32)

        return anchor_view1, anchor_view2, anchor_view3, anchor_view4, negatives_view1, negatives_view2, negatives_view3, negatives_view4
