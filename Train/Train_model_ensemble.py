def Train_model_ensemble(dataset, io_dir, train_params, num_levels, device='0'):
    
    import torch.nn as nn
    import os
    import time
    import numpy as np
    from model_inorm_mod import lpCNN
    import pytorch_ssim
    from tqdm import tqdm
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    from torch.autograd import Variable
    from SubDataset import SubDataset 
        
    seed = 2222
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    parent_read_dir = io_dir + 'pyramids/short_lbld/l2s_data_lp/'
    files = os.listdir(parent_read_dir)
    
    cuda1 = torch.device('cuda:' + device)
            
    num_epochs = train_params['num_epochs']
    lr_decay = train_params['lr_decay']
    lr = train_params['lr']
    bs = train_params['batch_size']
    save_freq = num_epochs-1
    ps = 48
    
    for file_ in tqdm(files):
     
        short_dir = parent_read_dir + file_ + '/'
        long_dir = io_dir + 'pyramids/long_lbld/lpyr/g' + str(num_levels) + '/'     
        save_dir = io_dir + 'checkpoints/lp_sub/model_ensemble/' + file_ + '/'
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        ## dataloader
        train_dataloader = DataLoader(SubDataset(short_dir, long_dir, dataset, ps), 
                                      batch_size=bs, shuffle=True)
        
        model = lpCNN()
        model.train()
        model = model.cuda(cuda1)
    
        l1_err = nn.L1Loss()
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=lr_decay, gamma=0.1)
        
        for epoch in range(num_epochs):
            
            scheduler.step(epoch)  # step to the learning rate in this epcoh
            
            epoch_loss = 0
            
            start_time = time.time()
            
            for n_count, batch_yx in enumerate(train_dataloader):
                
                optimizer.zero_grad()
                
                short_batch, long_batch = Variable(batch_yx[0]), Variable(batch_yx[1])
                
                out_batch = model(short_batch)
                                    
                l1_loss = l1_err(out_batch, long_batch)
                ssim_loss = pytorch_ssim.ssim(out_batch, long_batch)
                loss = l1_loss - ssim_loss            
                
                epoch_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            elapsed_time = time.time() - start_time
            # print('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
            np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
            if epoch%save_freq==0:
                torch.save(model, os.path.join(save_dir, 'modele.pth'))
       