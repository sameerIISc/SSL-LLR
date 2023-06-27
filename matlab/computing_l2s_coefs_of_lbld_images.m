function computing_l2s_coefs_of_lbld_images(dataset, io_dir)

    parent_dir = strcat(io_dir, 'img_data/train_lbld/'); 

    short_dir = strcat(parent_dir, 'short/');
    long_dir = strcat(parent_dir, 'long/');

    write_l2s_dir = strcat(parent_dir, 'l2s_coefs/');
    write_noise_dir = strcat(parent_dir, 'noise_vals/');

    mkdir(write_l2s_dir)
    mkdir(write_noise_dir)

    if strcmp(dataset, 'lol')
        files = dir(strcat(short_dir, '*.png'));
    else
        files = dir(strcat(short_dir, '*_00_*.png'));
    end
        
    for num_file = 1:length(files)

        name_short = files(num_file).name;
        im_short = im2double(imread(strcat(short_dir, name_short)));

        if strcmp(dataset, 'lol')
            im_long = im2double(imread(strcat(long_dir, name_short)));
        else
            long_file = dir(strcat(long_dir, name_short(1:5), '*.png'));
            im_long = im2double(imread(strcat(long_dir, long_file.name)));
        end

        l2s_coefs = compute_l2s_coefs(im_long, im_short);
        im_l2s = apply_l2s(im_long, l2s_coefs);

        noise = im_l2s - im_short;
        stdd(1) = std2(noise(:,:,1));
        stdd(2) = std2(noise(:,:,2));
        stdd(3) = std2(noise(:,:,3));

        save(strcat(write_l2s_dir, name_short(1:end-4), '.mat'), 'l2s_coefs');
        save(strcat(write_noise_dir, name_short(1:end-4), '.mat'), 'stdd');
    end