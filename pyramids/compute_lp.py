def compute_lp(read_dir, num_levels):
    
    from Build_GPyr import Build_GPyr
    import os
    from io_functions import read_img_as_tensor, save_tensor_as_img
    from tqdm import tqdm
    
    build_pyr = Build_GPyr()

    save_dir = read_dir[:-1] + '_lp/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    filenames = os.listdir(read_dir)
    
    for name in tqdm(filenames):
    
        img = read_img_as_tensor(read_dir + name)
        
        gpyr = build_pyr(img, num_levels)
        save_tensor_as_img(gpyr[-1], save_dir + name[:-4] + '.png')
