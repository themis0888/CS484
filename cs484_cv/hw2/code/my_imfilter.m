function output = my_imfilter(image, filter)
% This function is intended to behave like the built in function imfilter()
% when operating in convolution mode. See 'help imfilter'. 
% While "correlation" and "convolution" are both called filtering, 
% there is a difference. From 'help filter2':
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should meet the requirements laid out on the project webpage.

% Boundary handling can be tricky as the filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% we look at 'help imfilter', we see that there are several options to deal 
% with boundaries. 
% Please recreate the default behavior of imfilter:
% to pad the input image with zeros, and return a filtered image which matches 
% the input image resolution. 
% A better approach is to mirror or reflect the image content in the padding.

% Uncomment to call imfilter to see the desired behavior.
% output = imfilter(image, filter, 'conv');

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%
filter_size = size(filter);
if mod(filter_size(1),2) == 1
    ft_size = fix(size(filter)/2);
    im_size = size(image);
    B = zeros([ft_size(1)*2 + im_size(1) ft_size(2)*2 + im_size(2) 3]);
    for k = 1:3
        for i = 1:im_size(1)
            for j = 1:im_size(2) 
                B(i+ft_size(1), j+ft_size(2), k) = image(i,j,k);        
            end
        end
    end

    C = zeros(im_size);
    disp(size(image));
    disp(size(B));
    disp(size(C));
    filter_f = flipud(fliplr(filter));
    for k = 1:3
        for i = 1:im_size(1)
            for j = 1:im_size(2) 
                C(i,j,k) = sum(sum(B(i:i+2*ft_size(1):i,j:j+2*ft_size(2),k).*filter_f));
            end
        end
    end

    output = C;
else
    output = 'Error, put odd size filter';
end