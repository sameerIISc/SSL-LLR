def Train_sublp_with_pseudo_labels(dataset, train_params, io_dir, full_data_dir, device='0'):
    
    import torch.nn as nn
    import os
    import time
    from model_inorm_mod import lpCNN
    import pytorch_ssim
    from tqdm import tqdm
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    from torch.autograd import Variable
    from PLabel_Dataset import PLabel_Dataset
    from glob import glob
    from io_functions import read_img_as_tensor, save_tensor_as_img
    
    num_epochs = train_params['num_epochs_full']
    lr_decay = train_params['lr_decay_full']
    bs = train_params['batch_size']
    lr = train_params['lr']
    save_freq = num_epochs - 1
    ######### copying the real lbld data #################
    
    long_dir = io_dir + 'img_data/train_lbld/long_lp/' 
    short_dir = io_dir + 'img_data/train_lbld/short_lp/' 
    write_dir = io_dir + 'img_data/train_unlbld/pseudo_labels/'
    filenames = os.listdir(short_dir)
   
    for name in tqdm(filenames):
        if not dataset == 'lol':
            long_path = glob(long_dir + name[:5] + '*.png')  
            im_long = read_img_as_tensor(long_path[0], '0')
        else:
            long_path = long_dir + name
            im_long = read_img_as_tensor(long_path[0], '0')        
        save_tensor_as_img(im_long, write_dir + name)
    ###########################################################
    
    seed = 2222
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    short_dir = full_data_dir + 'train/short_lp/'
    long_dir = io_dir + 'img_data/train_unlbld/pseudo_labels/'     
    save_dir = io_dir + 'checkpoints/lp_sub/with_plabels/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cuda1 = torch.device('cuda:' + device)
        
    ps = 48
    
    ## dataloader
    train_dataloader = DataLoader(PLabel_Dataset(dataset, short_dir, long_dir, ps), 
                                  batch_size=bs, shuffle=True)
    
    model = lpCNN()
    model.train()
    model = model.cuda(cuda1)
    
    l1_err = nn.L1Loss()
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
        
            l1_loss = l1_err(out_batch, long_batch)
            ssim_loss = pytorch_ssim.ssim(out_batch, long_batch)
            loss = l1_loss - ssim_loss            
    
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        elapsed_time = time.time() - start_time
        # print('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        if epoch%save_freq==0:
            torch.save(model, os.path.join(save_dir, 'modelpl.pth'))