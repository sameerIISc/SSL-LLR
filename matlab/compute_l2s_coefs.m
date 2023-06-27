function coefs = compute_l2s_coefs(im_long, im_short)

    degree=3;

    for j = 1:3
        short = im_short(:,:,j);
        long = im_long(:,:,j);    
        coefs(j,:) = polyfit(long(:), short(:), degree);
    end
    
end