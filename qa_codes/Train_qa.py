def Train_qa(dataset, split, device, io_dir):

    import os
    import time
    import numpy as np
    from model import qfcnn
    from glob import glob
    from tqdm import tqdm
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    from torch.autograd import Variable
    import cv2
    
    seed = 2222
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    img_dir = io_dir + 'img_data/train_unlbld/short_lp/'
    prtrbtn_dir = io_dir + 'img_data/train_unlbld/model_prtrbd/with_5/'  
    save_dir = io_dir + 'checkpoints/contrast_learning/with_5/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cuda1 = torch.device('cuda:' + device)
    save_freq = 10
    
    ## creating the dataset
    class SIDDataset(Dataset):
        def __init__(self, img_dir):
           
            self.files_img = os.listdir(img_dir)
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
            
            prtrbd_paths = glob(prtrbtn_dir + name_img[:-4] + '_*.npy')
           
            # print prtrbd_paths
            num_anchor = np.random.randint(len(prtrbd_paths))
            
            anchor_path = prtrbd_paths[num_anchor]
            im_anchor = self.read_data(anchor_path)
            
            H = im_anchor.shape[1]
            W = im_anchor.shape[2]
            
            ps_x = H/2
            ps_y = W/2
            
            ######## reading view 1 of anchor image ###########
            
            num_view1_patch = np.random.randint(4)
            
            num_view1_patch_x = num_view1_patch/2
            num_view1_patch_y = np.mod(num_view1_patch,2)
            
            view1_patch_x = num_view1_patch_x*ps_x
            view1_patch_y = num_view1_patch_y*ps_y
            
            anchor_view1 = im_anchor[:3, view1_patch_x:view1_patch_x+ps_x,view1_patch_y:view1_patch_y+ps_y] 
            
            ######## reading view 2 of anchor image ###########
            
            num_view2_patch = np.random.randint(4)
            
            while num_view2_patch == num_view1_patch:
                num_view2_patch = np.random.randint(4)
            
            num_view2_patch_x = num_view2_patch/2
            num_view2_patch_y = np.mod(num_view2_patch,2)
            
            view2_patch_x = num_view2_patch_x*ps_x
            view2_patch_y = num_view2_patch_y*ps_y
            
            anchor_view2 = im_anchor[:3, view2_patch_x:view2_patch_x+ps_x,view2_patch_y:view2_patch_y+ps_y] 
            
            ######## reading negatives ##########
            negatives_view1 = np.zeros((len(prtrbd_paths)-1,3,ps_x,ps_y), dtype=np.float32)
            negatives_view2 = np.zeros((len(prtrbd_paths)-1,3,ps_x,ps_y), dtype=np.float32)
            count = 0
            for path in prtrbd_paths:
                if not path == anchor_path:
                    img = self.read_data(path)
                    negatives_view1[count, :,:,:] = img[:3, view1_patch_x:view1_patch_x+ps_x,view1_patch_y:view1_patch_y+ps_y] 
                    negatives_view2[count, :,:,:] = img[:3, view2_patch_x:view2_patch_x+ps_x,view2_patch_y:view2_patch_y+ps_y] 
                  
                    count += 1
            
            anchor_view1 = torch.tensor(anchor_view1.copy(), device=cuda1, dtype=torch.float32)
            anchor_view2 = torch.tensor(anchor_view2.copy(), device=cuda1, dtype=torch.float32)
            negatives_view1 = torch.tensor(negatives_view1.copy(), device=cuda1, dtype=torch.float32)
            negatives_view2 = torch.tensor(negatives_view2.copy(), device=cuda1, dtype=torch.float32)
            
            return anchor_view1, anchor_view2, negatives_view1, negatives_view2
    
    ## dataloader
    train_dataloader = DataLoader(SIDDataset(img_dir), 
                                  batch_size=64, shuffle=True)
    
    model = qfcnn()
    model.train()
    model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    
    for epoch in tqdm(range(21)):
        
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        
        epoch_loss = 0
        
        start_time = time.time()
        
        for n_count, batch_yx in enumerate(train_dataloader):
                    
            optimizer.zero_grad()
            
            anchor_view1, anchor_view2, negatives_view1, negatives_view2 = Variable(batch_yx[0]), Variable(batch_yx[1]), Variable(batch_yx[2]), Variable(batch_yx[3])
                
            tau = 0.1
            
            ######################## computing and normalizing anchor features both views  ####################
            
            anchor_view1_features = model(anchor_view1)
            anchor_view2_features = model(anchor_view2)
            
            anchor_view1_features = anchor_view1_features/(torch.norm(anchor_view1_features, dim=1).unsqueeze(dim=1))
            anchor_view2_features = anchor_view2_features/(torch.norm(anchor_view2_features, dim=1).unsqueeze(dim=1))
       
            anchor_view1_features = anchor_view1_features.unsqueeze(dim=0).permute(1,0,2)
            anchor_view2_features = anchor_view2_features.unsqueeze(dim=0).permute(1,0,2)
    
            numerator = torch.exp(torch.matmul(anchor_view1_features, anchor_view2_features.permute(0,2,1))/tau)
            numerator = numerator[:,0:1,0]
            
            ######################## computing negative view features ############################
        
            batch_size, num_prtrbtns, c, h, w = negatives_view1.size()
            negatives_view = torch.cat((negatives_view1, negatives_view2), dim=1)
            
            negatives_view_features = torch.zeros((batch_size,2*num_prtrbtns,128), device=cuda1, dtype=torch.float32)
            for batch_num in range(batch_size):
                temp = model(negatives_view[batch_num, :,:,:,:])
                temp = temp/(torch.norm(temp, dim=1).unsqueeze(dim=1))    
                negatives_view_features[batch_num,:,:] = temp
            
            denominator1_int = torch.matmul(anchor_view1_features, negatives_view_features.permute(0,2,1))
            denominator2_int = torch.matmul(anchor_view2_features, negatives_view_features.permute(0,2,1))
          
            denominator1 = torch.sum(torch.exp(denominator1_int/tau), dim=2) + numerator
            denominator2 = torch.sum(torch.exp(denominator2_int/tau), dim=2) + numerator
      
            ######################## computing the loss and mean ################################
        
            view1_loss = torch.mean(torch.log(denominator1) - torch.log(numerator))
            view2_loss = torch.mean(torch.log(denominator2) - torch.log(numerator))
            
            loss = view1_loss + view2_loss
            
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
        
        elapsed_time = time.time() - start_time
        # print('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.8f')
        if epoch%save_freq==0:
            torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))