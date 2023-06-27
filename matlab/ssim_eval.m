function ssim_eval(dataset, method, io_dir)

    main_dir = '/mnt/e4e2203b-ecf7-4807-b641-1fa71921092d/';
    
    enh_dir = strcat(io_dir, 'restored/', method);
    long_dir = strcat(main_dir, 'Datasets/', dataset, '/test/long/');

    files = dir(strcat(enh_dir, '*.png'));
%     length(files)
    for i = 1:length(files)
%         i 
        name = files(i).name;
        im_enh = imread(strcat(enh_dir, name));

        if strcmp(dataset, 'lol')
            im_long = imread(strcat(long_dir, name));
        else    
            long_file = dir(strcat(long_dir, name(1:5), '*.png'));
            im_long = imread(strcat(long_dir, long_file.name));
        end

        sim(i) = ssim(im_enh, im_long);
        pnr(i) = psnr(im_enh, im_long);
    end  
    fprintf('\nThe ssim score is: %f', mean(sim))
    fprintf('\nThe psnr score is: %f\n', mean(pnr))
    