clear all
clc

dataset = 'sony';
frac_lbld = '5%_data';

for num_split = 1%:10
    num_split
    split = strcat(num2str(num_split), '/');
    io_dir = strcat('../../', dataset, '/', frac_lbld, '/split', split);
    
    %% 1st step. To be run after 1st step of python code
%     fprintf('******* computing the coefs ********* \n');
%     computing_l2s_coefs_of_lbld_images(dataset, io_dir);
%     fprintf('******* generating l2s data ********* \n');
%     transforming_gt_images(dataset, io_dir);
%     fprintf('******* copying real to synthetic ********* \n');
%     copying_real_to_l2s(dataset, io_dir);

    %% 2nd step. To be run after 2nd step of python code
    compute_plabels(dataset, io_dir);
end    