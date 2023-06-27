def Train_BandpassCNN_lbld(dataset, io_dir, train_params, num_levels, device='0'):
 
    import os
    import time
    import numpy as np
    from model_subCNN import SubCNN
    from tqdm import tqdm
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    from torch.autograd import Variable
    import torch.nn as nn
    from SubDataset import SubDataset
    
    num_epochs = train_params['num_epochs']
    lr_decay = train_params['lr_decay']
    lr = train_params['lr']
    bs = train_params['batch_size']
    save_freq = num_epochs-1
    
    subs_ps = []
    for l in range(num_levels-2):
        subs_ps.append(('l' + str(l+1) + '/', pow(2,num_levels+3-l)))
    subs_ps.append(('l' + str(l+2) + '/', pow(2,num_levels+3-l)))

    for sub, patch_size in subs_ps:
                
        print(' \n ######### Training on sub ' + sub[:2] + ' ########### \n') 
        
        short_dir = io_dir + 'pyramids/short_lbld/gpyr/' + sub   
        long_dir = io_dir + 'pyramids/long_lbld/lpyr/' + sub
        
        save_dir = io_dir + 'checkpoints/bsubs/g2l/' + sub
    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        cuda1 = torch.device('cuda:' + device)
            
        ## dataloader
        train_dataloader = DataLoader(SubDataset(short_dir, long_dir, dataset, patch_size), 
                                      batch_size=bs, shuffle=True)
        
        model = SubCNN()
        model.train()
        model = model.cuda(cuda1)
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=lr_decay, gamma=0.1)
        
        for epoch in tqdm(range(num_epochs)):
            
            scheduler.step(epoch)  # step to the learning rate in this epcoh
            
            epoch_loss = 0
            
            start_time = time.time()
            
            for n_count, batch_yx in enumerate(train_dataloader):
                
                optimizer.zero_grad()
                short_batch, long_batch = Variable(batch_yx[0]), Variable(batch_yx[1])
                out_batch = model(short_batch)
                loss = criterion(out_batch, long_batch)
                epoch_loss += loss.item()
                optimizer.step()
                
            elapsed_time = time.time() - start_time
            # print('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
            # np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
            if epoch%save_freq==0:
                torch.save(model, os.path.join(save_dir, 'modelbp.pth'))