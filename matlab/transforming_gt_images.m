function transforming_gt_images(dataset, io_dir)

    parent_dir = strcat(io_dir, 'img_data/train_lbld/');

    %% defining the paths
    l2s_dir = strcat(parent_dir, 'l2s_coefs/');
    noise_dir = strcat(parent_dir, 'noise_vals/');

    target_dir = strcat(parent_dir, 'long/');

    parent_write_dir = strcat(parent_dir, 'l2s_data/with_5_folders/');
    mkdir(parent_write_dir)

    source_files = dir(strcat(l2s_dir, '*.mat'));
    target_files = dir(strcat(target_dir, '*.png'));

    %% applying the transforms

    if strcmp(dataset, 'lol')
        num_images = 1;
    else
        num_images = 10;
    end
    
    f = waitbar(0, 'Starting');
    n = length(source_files);
    
    for num_source = 1:length(source_files)
        waitbar(num_source/n, f, sprintf('Selecting pseudo label: %d %%', floor(num_source/n*100)));
        pause(0.1);
        
        name_source = source_files(num_source).name;

        load(strcat(l2s_dir, name_source));
        load(strcat(noise_dir, name_source));

        write_dir = strcat(parent_write_dir, name_source(1:end-4), '/');
        mkdir(write_dir)
        for num_target = 1:length(target_files)
            target_name = target_files(num_target).name;
            im_target = im2double(imread(strcat(target_dir, target_name)));
            im_l2s = apply_l2s(im_target, l2s_coefs);   
            for num = 1:num_images
                gnoise = randn(size(im_target)); 
                for ch = 1:3
                    fake_noise(:,:,ch) = stdd(ch)*gnoise(:,:,ch);
                end
                imwrite(im_l2s+fake_noise, strcat(write_dir, target_name(1:end-4), '_', num2str(num), '.png'));
            end
        end
    end
    close(f);