function exposure = compute_exposure(name)

    for num_letter = 1:length(name)
        letter = name(num_letter);
        if letter == 's'
            exp = name(num_letter-1);

            if exp == '3'
                exposure = '0.033';
            elseif exp == '4'
                exposure = '0.04';
            else
                exposure = '0.1';                
            end 
            break
        end
    end