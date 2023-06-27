def test_qa(io_dir, device='0'):
    
    import os
    from io_functions import read_img_as_tensor
    from torch.autograd import Variable
    from scipy.io import savemat
    import torch
    from glob import glob
    from tqdm import tqdm
    
    cuda1 = torch.device('cuda:' + device)
    
    enh_dir = io_dir + 'img_data/train_unlbld/restored_with_ensemble/'
    long_dir = io_dir + 'img_data/train_lbld/long_lp/'
    
    pred_qlty_dir = io_dir + 'img_data/train_unlbld/pred_qlty_mv/'
    model_dir = io_dir + 'checkpoints/qa_model_mv/modelq.pth'

    if not os.path.exists(pred_qlty_dir):
        os.mkdir(pred_qlty_dir)
    
    enh_files = glob(enh_dir + '*.png')
    long_files = os.listdir(long_dir)
    
    model = torch.load(model_dir)
    model = model.eval()
    model = model.cuda(cuda1)
    
    #################### computing long_features ###########################
    sum_long_features = torch.zeros((1,128), device=cuda1, dtype=torch.float32)
    for name in tqdm(long_files):

        im_long = read_img_as_tensor(long_dir + name, device)
        
        with torch.no_grad():
            long_features = model(im_long)
    
        sum_long_features += long_features
    
    sum_long_features = sum_long_features/len(long_files)
    sum_long_features = sum_long_features/torch.norm(sum_long_features)
    
    #################### computing similarity of short_features with long_features ###########################
    for path in tqdm(enh_files):
        
        name = os.path.basename(path)
          
        im_enh = read_img_as_tensor(enh_dir + name, device)
        
        with torch.no_grad():
            enh_features = model(im_enh)
        
        enh_features = enh_features/torch.norm(enh_features)
        
        pred_qlty = torch.matmul(enh_features, sum_long_features.t())
        pred_qlty = Variable(pred_qlty, requires_grad=False).cpu().numpy()
        pred_qlty_dict = {'pred_qlty':pred_qlty}
        savemat(pred_qlty_dir + name[:-4] + '.mat', pred_qlty_dict) 