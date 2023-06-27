def Train_qa_mv(dataset, io_dir, train_params, device='0'):
    
    import torch.nn as nn
    import os
    import time
    import numpy as np
    from qa_model import qfcnn
    
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR
    from torch.autograd import Variable
    from ContrastiveQA_dataset import ContrastiveQA_dataset    
    from tqdm import tqdm
    
    seed = 2222
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    img_dir = io_dir + 'img_data/train_unlbld/short_lp/'
    prtrbtn_dir = io_dir + 'img_data/train_unlbld/restored_with_ensemble/'  
    save_dir = io_dir + 'checkpoints/qa_model_mv/'
    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cuda1 = torch.device('cuda:' + device)
    
    bs = train_params['batch_size_qa']
    lr = train_params['lr_qa']
    num_epochs = train_params['num_epochs_qa'] 
    tau = train_params['temperature_qa']
    save_freq = num_epochs-1
    
    ## dataloader
    train_dataloader = DataLoader(ContrastiveQA_dataset(img_dir, prtrbtn_dir, dataset), 
                                  batch_size=bs, shuffle=True)
    
    model = qfcnn()
    model.train()
    model = model.cuda(cuda1)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1)
    scheduler = MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    
    for epoch in tqdm(range(num_epochs)):
        
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        
        epoch_loss = 0
        
        start_time = time.time()
        
        for n_count, batch_yx in enumerate(train_dataloader):
            
            optimizer.zero_grad()
            
            anchor_view1, anchor_view2, anchor_view3, anchor_view4, negatives_view1, negatives_view2, negatives_view3, negatives_view4  = Variable(batch_yx[0]), Variable(batch_yx[1]), Variable(batch_yx[2]), Variable(batch_yx[3]), Variable(batch_yx[4]), Variable(batch_yx[5]), Variable(batch_yx[6]), Variable(batch_yx[7])
            
            negatives_view = torch.cat((negatives_view1, negatives_view2, negatives_view3, negatives_view4), dim=1)
            
            anchor_view1_features = model(anchor_view1)
            anchor_view2_features = model(anchor_view2)
            anchor_view3_features = model(anchor_view3)
            anchor_view4_features = model(anchor_view4)
            
            anchor_view1_features = anchor_view1_features/(torch.norm(anchor_view1_features, dim=1).unsqueeze(dim=1))
            anchor_view2_features = anchor_view2_features/(torch.norm(anchor_view2_features, dim=1).unsqueeze(dim=1))
            anchor_view3_features = anchor_view3_features/(torch.norm(anchor_view3_features, dim=1).unsqueeze(dim=1))
            anchor_view4_features = anchor_view4_features/(torch.norm(anchor_view4_features, dim=1).unsqueeze(dim=1))
       
            anchor_view1_features = anchor_view1_features.unsqueeze(dim=0).permute(1,0,2)
            anchor_view2_features = anchor_view2_features.unsqueeze(dim=0).permute(1,0,2)
            anchor_view3_features = anchor_view3_features.unsqueeze(dim=0).permute(1,0,2)
            anchor_view4_features = anchor_view4_features.unsqueeze(dim=0).permute(1,0,2)
        
            batch_size, num_prtrbtns, c, h, w = negatives_view1.size()
            
            negatives_view_features = torch.zeros((batch_size,4*num_prtrbtns,128), device=cuda1, dtype=torch.float32)
            for batch_num in range(batch_size):
                temp = model(negatives_view[batch_num, :,:,:,:])
                temp = temp/(torch.norm(temp, dim=1).unsqueeze(dim=1))    
                negatives_view_features[batch_num,:,:] = temp
                
            numerator_12 = torch.exp(torch.matmul(anchor_view1_features, anchor_view2_features.permute(0,2,1))/tau)
            numerator_12 = numerator_12[:,0:1,0]
            
            numerator_13 = torch.exp(torch.matmul(anchor_view1_features, anchor_view3_features.permute(0,2,1))/tau)
            numerator_13 = numerator_13[:,0:1,0]
            
            numerator_14 = torch.exp(torch.matmul(anchor_view1_features, anchor_view4_features.permute(0,2,1))/tau)
            numerator_14 = numerator_14[:,0:1,0]
            
            numerator_23 = torch.exp(torch.matmul(anchor_view2_features, anchor_view3_features.permute(0,2,1))/tau)
            numerator_23 = numerator_23[:,0:1,0]
            
            numerator_24 = torch.exp(torch.matmul(anchor_view2_features, anchor_view4_features.permute(0,2,1))/tau)
            numerator_24 = numerator_24[:,0:1,0]
            
            numerator_34 = torch.exp(torch.matmul(anchor_view3_features, anchor_view4_features.permute(0,2,1))/tau)
            numerator_34 = numerator_34[:,0:1,0]
            
            denominator1_int = torch.matmul(anchor_view1_features, negatives_view_features.permute(0,2,1))
            denominator2_int = torch.matmul(anchor_view2_features, negatives_view_features.permute(0,2,1))
            denominator3_int = torch.matmul(anchor_view3_features, negatives_view_features.permute(0,2,1))
            denominator4_int = torch.matmul(anchor_view4_features, negatives_view_features.permute(0,2,1))
          
            denominator12 = torch.sum(torch.exp(denominator1_int/tau), dim=2) + numerator_12
            denominator13 = torch.sum(torch.exp(denominator1_int/tau), dim=2) + numerator_13
            denominator14 = torch.sum(torch.exp(denominator1_int/tau), dim=2) + numerator_14
            denominator21 = torch.sum(torch.exp(denominator2_int/tau), dim=2) + numerator_12
            denominator23 = torch.sum(torch.exp(denominator2_int/tau), dim=2) + numerator_23
            denominator24 = torch.sum(torch.exp(denominator2_int/tau), dim=2) + numerator_24
            denominator31 = torch.sum(torch.exp(denominator3_int/tau), dim=2) + numerator_13
            denominator32 = torch.sum(torch.exp(denominator3_int/tau), dim=2) + numerator_23
            denominator34 = torch.sum(torch.exp(denominator3_int/tau), dim=2) + numerator_34
            denominator41 = torch.sum(torch.exp(denominator4_int/tau), dim=2) + numerator_14
            denominator42 = torch.sum(torch.exp(denominator4_int/tau), dim=2) + numerator_24
            denominator43 = torch.sum(torch.exp(denominator4_int/tau), dim=2) + numerator_34
           
            loss12 = torch.mean(torch.log(denominator12) - torch.log(numerator_12))
            loss13 = torch.mean(torch.log(denominator13) - torch.log(numerator_13))
            loss14 = torch.mean(torch.log(denominator14) - torch.log(numerator_14))
            loss21 = torch.mean(torch.log(denominator21) - torch.log(numerator_12))
            loss23 = torch.mean(torch.log(denominator23) - torch.log(numerator_23))
            loss24 = torch.mean(torch.log(denominator24) - torch.log(numerator_24))
            loss31 = torch.mean(torch.log(denominator31) - torch.log(numerator_13))
            loss32 = torch.mean(torch.log(denominator32) - torch.log(numerator_23))
            loss34 = torch.mean(torch.log(denominator34) - torch.log(numerator_34))
            loss41 = torch.mean(torch.log(denominator41) - torch.log(numerator_14))
            loss42 = torch.mean(torch.log(denominator42) - torch.log(numerator_24))
            loss43 = torch.mean(torch.log(denominator43) - torch.log(numerator_34))
            
            loss = loss12 + loss13 + loss14 + loss21 + loss23 + loss24 + loss31 + loss32 + loss34 + loss41 + loss42 + loss43 
            
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
        
        elapsed_time = time.time() - start_time
        # print('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.8f')
        if epoch%save_freq==0:
            torch.save(model, os.path.join(save_dir, 'modelq.pth'))