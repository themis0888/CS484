
% PRINCIPAL PROGRAM
warning off
% Clear all
clc, close all, clear all
letter_list_t = 'ABCDE';
for ty = 1:5
    % Read image
    imagen=imread(sprintf('DATA/VALID_DATA/Char%s.bmp',letter_list_t(ty)));
    % Show image
    figure(1);
    imshow(imagen);
    title('INPUT IMAGE WITH NOISE')
    % Convert to gray scale
    if size(imagen,3)==3 %RGB image
        imagen=rgb2gray(imagen);
    end
    % Convert to BW
    threshold = 0.9;
    %threshold = graythresh(imagen);
    imagen = ~imbinarize(imagen,threshold);
    % Remove all object containing fewer than 30 pixels
    imagen = bwareaopen(imagen, 25);
    %Storage matrix word from image
    word=[ ];
    re=imagen;
    %Opens text.txt as file for write
    %fid = fopen('text.txt', 'wt');
    % Load templates
    load template
    global template
    % Compute the number of letters in template file
    num_letras=size(template,2);

    imgn=re;
    %-----------------------------------------------------------------
    % Label and count connected components
    [L, Ne] = bwlabel(imgn);
    dim = [.2 .5 .3 .3];
    for n=1:Ne
        [r,c] = find(L==n);
        % Extract letter
        n1=imgn(min(r):max(r),min(c):max(c));
        % Resize letter (same size of template)
        
        %Uncomment line below to see letters one by one
        %imshow(img_r);pause(0.5)
        %-------------------------------------------------------------------
        % Call fcn to convert image to text
        letter = [];
        tmp = [];
        for angle = 0:180:360
            n1_tmp = imrotate(n1, angle);
            [rr,cc] = find(n1_tmp==1);
            n1_tmp = n1_tmp(min(rr):max(rr),min(cc):max(cc));
            img_r = imresize(n1_tmp,[200 200]);
            [letter(end+1), tt] = read_letter(img_r,num_letras);
            tmp = [tmp;tt];
            figure(2)
            imshow(img_r)
            title(sprintf('%s',letter(end)))
        end
        
        ttt=ceil(find(max(max(tmp))==max(tmp))/60)+96;
        figure(1)
        rectangle('Position',[min(c) min(r) max(c)-min(c) max(r)-min(r)])
        text((max(c) + min(c))/2-20, min(r) - 50,sprintf('%s',ttt))

    end
    dirpath = 'result';
    if dirpath(end) ~= '/', dirpath = [dirpath '/']; end
    if (exist(dirpath, 'dir') == 0), mkdir(dirpath); end
    saveas(gcf,sprintf('result/test_%s.bmp',letter_list_t(ty)))

end

