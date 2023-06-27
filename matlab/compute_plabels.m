function compute_plabels(dataset, io_dir)

    parent_dir = strcat(io_dir, 'img_data/train_unlbld/');

    restored_version_dir = strcat(parent_dir, 'restored_with_ensemble/');
    qlty_dir = strcat(parent_dir, 'pred_qlty_mv/');
    pretrained_dir = strcat(parent_dir, 'lp_pretrained/');
    write_dir = strcat(parent_dir, 'pseudo_labels/');
    mkdir(write_dir)

    files = dir(strcat(pretrained_dir, '*.png'));
    
    f = waitbar(0, 'Starting');
    n = length(files);
    
    for i = 1:n
        waitbar(i/n, f, sprintf('Selecting pseudo label: %d %%', floor(i/n*100)));
        pause(0.1);
        name = files(i).name;
        rv_files = dir(strcat(restored_version_dir, name(1:end-4), '_*.png'));
        im_ptrained = imread(strcat(pretrained_dir, name));
        clear sim
        for num_rv = 1:length(rv_files)
            name_rv = rv_files(num_rv).name;
            im_rv = imread(strcat(restored_version_dir, name_rv));
            sim_pt = ssim(im_rv, im_ptrained);
            load(strcat(qlty_dir, name_rv(1:end-4), '.mat'));
            pred_qltys(num_rv) = pred_qlty + sim_pt;
        end
        [~, locq] =  max(pred_qltys);
        im_best = imread(strcat(restored_version_dir, rv_files(locq).name));
        imwrite(im_best, strcat(write_dir, name));
    end  
    close(f);