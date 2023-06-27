def test_LowpassCNN_pt_unlbld(dataset, io_dir, device='0'):

    import sys
    
    sys.path.insert(0, '../models/')
    sys.path.insert(0, '../utils/')
    
    import os
    from io_functions import read_img_as_tensor, save_tensor_as_img
    from tqdm import tqdm
    import torch
    
    cuda1 = torch.device('cuda:' + device)
    
    short_dir = io_dir + 'img_data/train_unlbld/short_lp/'
    enh_dir = io_dir + 'img_data/train_unlbld/lp_pretrained/'
    
    lp_model_path = io_dir + 'checkpoints/lp_sub/only_lbld/modellp_pt.pth'
    
    if not os.path.exists(enh_dir):
        os.makedirs(enh_dir)
    
    filenames = os.listdir(short_dir)
    
    modellp = torch.load(lp_model_path)
    modellp.eval()
    modellp = modellp.cuda(cuda1)   
    
    for name in tqdm(filenames):
            
        im_short = read_img_as_tensor(short_dir + name, device)
        
        with torch.no_grad():
            im_enh = modellp(im_short)
    
        save_tensor_as_img(im_enh, enh_dir + name)