def compute_lp_ensemble_data(dataset, io_dir, num_levels, device='0'):

    import sys
    sys.path.insert(0, '../utils/')
    
    from Build_GPyr import Build_GPyr
    import torch
    import os
    from io_functions import read_img_as_tensor, save_tensor_as_npy
    
    cuda1 = torch.device('cuda:' + device)
    
    build_pyr = Build_GPyr(device)
    
    parent_read_dir = io_dir + 'img_data/train_lbld/l2s_data/with_5_folders/'
    parent_save_dir = io_dir + 'pyramids/short_lbld/l2s_data_lp/'
    
    if not os.path.exists(parent_save_dir):
        os.mkdir(parent_save_dir)
    
    parent_filenames = os.listdir(parent_read_dir)
    
    for count, parent_name in enumerate(parent_filenames):
    
        read_dir = parent_read_dir + parent_name + '/'
        save_dir = parent_save_dir + parent_name + '/'
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        filenames = os.listdir(read_dir)
        
        for name in filenames:
            img = read_img_as_tensor(read_dir + name, device)
            gpyr = build_pyr(img, num_levels)
            save_tensor_as_npy(gpyr[-1], save_dir + name[:-4] + '.npy')