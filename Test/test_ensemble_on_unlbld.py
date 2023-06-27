def test_ensemble_unlbld(io_dir, device='0'):

    import sys
    
    sys.path.insert(0, '../models/')
    sys.path.insert(0, '../utils/')
    
    from Build_GPyr_1 import Build_GPyr
    import os
    from io_functions import read_img_as_tensor, save_tensor_as_img
    from tqdm import tqdm
    import torch
    
    build_pyr = Build_GPyr(device)

    cuda1 = torch.device('cuda:' + device)
    
    short_dir = io_dir + 'img_data/train_unlbld/short_lp/'
    enh_dir = io_dir + 'img_data/train_unlbld/restored_with_ensemble/'
    
    parent_model_dir = io_dir + 'checkpoints/lp_sub/model_ensemble/'
    
    model_dirs = os.listdir(parent_model_dir)
    model_dirs = sorted(model_dirs)

    if not os.path.exists(enh_dir):
        os.makedirs(enh_dir)
    
    filenames = os.listdir(short_dir)
    
    for name in tqdm(filenames):
     
        im_short = read_img_as_tensor(short_dir + name, device)
            
        for model_num, model_dir in enumerate(model_dirs):
            
            modellp = torch.load(parent_model_dir + model_dir + '/modele.pth')
            modellp.eval()
            modellp = modellp.cuda(cuda1)   
        
            with torch.no_grad():
                im_enh = modellp(im_short)
            
            save_tensor_as_img(im_enh, enh_dir + name[:-4] + '_' + str(model_num) + '.png')