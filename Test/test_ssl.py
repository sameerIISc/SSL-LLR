def test_ssl(io_dir, full_data_dir, num_levels, device='0'):
    
    from Build_GPyr import Build_GPyr
    import os
    from LPyrRecon import reconLPyr
    from io_functions import read_img_as_tensor, save_tensor_as_img
    from tqdm import tqdm
    import torch

    cuda1 = torch.device('cuda:' + device)
  
    reconstruct = reconLPyr()

    short_dir = full_data_dir + 'test/short/'
    enh_dir = io_dir + 'img_data/restored/with_ssl/' 
    
    if not os.path.exists(enh_dir):
        os.makedirs(enh_dir)
    
    filenames = os.listdir(short_dir)
    
    bpcnn_model_dir = io_dir + 'checkpoints/bsubs/g2l/' 
    
    models = []
    for l in range(1,num_levels):
        models.append(torch.load(bpcnn_model_dir + 'l' + str(l) + '/modelbp.pth').eval().cuda(cuda1)) 
      
    modellp = torch.load(io_dir + 'checkpoints/lp_sub/with_plabels/modelpl.pth').eval().cuda(cuda1)
    build_pyr = Build_GPyr()

    for name in tqdm(filenames):
                
        im_short = read_img_as_tensor(short_dir + name)
        pyr = build_pyr(im_short, num_levels)
        
        subs_out = []
        with torch.no_grad():
            for l in range(num_levels-1):
                sub_in = pyr[l]
                subs_out.append(models[l](sub_in))
            subs_out.append(modellp(pyr[-1]))
       
        im_enh = reconstruct(subs_out)
        save_tensor_as_img(im_enh, enh_dir + name)