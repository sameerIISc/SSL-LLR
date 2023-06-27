function im_l2s = apply_l2s(img, coefs)

    number_of_coefs = 4;

    im_l2s = zeros(size(img));

    for j = 1:number_of_coefs
        im_l2s(:,:,1) = im_l2s(:,:,1) + (img(:,:,1).^(number_of_coefs-j))*coefs(1,j);
        im_l2s(:,:,2) = im_l2s(:,:,2) + (img(:,:,2).^(number_of_coefs-j))*coefs(2,j);
        im_l2s(:,:,3) = im_l2s(:,:,3) + (img(:,:,3).^(number_of_coefs-j))*coefs(3,j);
    end
end