function copying_real_to_l2s(dataset, io_dir)

    parent_dir = strcat(io_dir, 'img_data/train_lbld/');
    read_dir = strcat(parent_dir, 'short/');

    parent_write_dir =strcat(parent_dir,  'l2s_data/with_5_folders/');
    parent_folders = dir(strcat(parent_write_dir, '*s'));

    img_files = dir(strcat(read_dir, '*.png'));

    for num_folder = 1:length(parent_folders)
        num_folder
        folder_name = parent_folders(num_folder).name;
        write_dir = strcat(parent_write_dir, folder_name, '/');

        id_ = folder_name(1:5);
        exposure = compute_exposure(folder_name);

        img_files = dir(strcat(read_dir, id_, '_*_', exposure, 's.png'));

        for num_file = 1:length(img_files)
            img_name = img_files(num_file).name;
            img = imread(strcat(read_dir, img_name));
            imwrite(img, strcat(write_dir, img_name));
        end

        if length(img_files)<=2
            write_id = 2;
            for num_file = 1:length(img_files)
                img_name = img_files(num_file).name;
                img = imread(strcat(read_dir, img_name));
                for num_write = 1:3
                    write_name = strcat(id_, '_0', num2str(write_id), '_', exposure, 's.png');  
                    write_id = write_id+1;
                    imwrite(img, strcat(write_dir, write_name));
                end
            end
        end
    end