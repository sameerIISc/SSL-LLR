def compute_lp_unlbld(io_dir, num_levels):
    
    import sys
    sys.path.insert(0, '../utils/')
    
    from Build_GPyr import Build_GPyr
    import torch
    import os
    from io_functions import read_img_as_tensor, save_tensor_as_img
    from tqdm import tqdm
    
    build_pyr = Build_GPyr()

    read_dir = io_dir + 'img_data/train_unlbld/short/'
    save_dir = io_dir + 'img_data/train_unlbld/short_lp/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    filenames = os.listdir(read_dir)
    
    for name in tqdm(filenames):
    
        img = read_img_as_tensor(read_dir + name)
        
        gpyr = build_pyr(img, num_levels)
        save_tensor_as_img(gpyr[-1], save_dir + name[:-4] + '.png')
