clear all
clc

dataset = 'sony';
main_dir = '/mnt/e4e2203b-ecf7-4807-b641-1fa71921092d/';
io_dir = strcat(main_dir,  'Instance_Norm/', dataset, '/5%_data/');

short_dir = strcat(io_dir, 'img_data/train_lbld/short_lp_subset/');
long_dir = strcat(io_dir, 'img_data/train_lbld/long_lp/');

short_files = dir(strcat(short_dir, '*.png'));

write_dir = strcat(io_dir, 'img_data/train_lbld/long_lp_ex/');
mkdir(write_dir)
for i = 1:length(short_files)
    
    name_short = short_files(i).name;
    
    long_file = dir(strcat(long_dir, name_short(1:5), '*.png'));
    im_long = imread(strcat(long_dir, long_file.name));
    
    imwrite(im_long, strcat(write_dir, name_short));
end