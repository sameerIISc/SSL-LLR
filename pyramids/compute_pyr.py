def compute_pyr(dataset, exposure, io_dir, num_levels):
    
    from Build_LPyr import Build_LPyr
    from Build_GPyr import Build_GPyr
    import torch
    import os
    from io_functions import read_img_as_tensor, save_tensor_as_npy, save_tensor_as_img
    from tqdm import tqdm
    
    if exposure == 'long':
        parent_dir = 'lpyr/'
        build_pyr = Build_LPyr()
    else:
        parent_dir = 'gpyr/'
        build_pyr = Build_GPyr()
    
    read_dir = io_dir + 'img_data/train_lbld/' + exposure + '/'
    save_dir = io_dir + 'pyramids/' + exposure + '_lbld/' + parent_dir
    save_img_dir = io_dir + 'img_data/train_lbld/' + exposure + '_lp/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        for l in range(num_levels-1):
            os.mkdir(save_dir + 'l' + str(l+1) + '/')
        os.mkdir(save_dir + 'g' + str(num_levels) + '/')
        os.mkdir(save_img_dir)
  
    filenames = os.listdir(read_dir)    
    for name in tqdm(filenames):
        img = read_img_as_tensor(read_dir + name)
        pyr = build_pyr(img, num_levels)
        for l in range(num_levels-1):
            save_tensor_as_npy(pyr[l], save_dir + 'l' + str(l+1) + '/' + name[:-4] + '.npy')
        save_tensor_as_npy(pyr[-1], save_dir + 'g' + str(len(pyr)) + '/' + name[:-4] + '.npy')
        save_tensor_as_img(pyr[-1], save_img_dir + name[:-4] + '.png')
