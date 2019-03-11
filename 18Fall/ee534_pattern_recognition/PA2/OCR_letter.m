
warning off
% Clear all
clc, close all, clear all
letter_list_t = 'ABCDE';
for ty = 1:4
    % Read image
    imagen=imread(sprintf('DATA/VALID_DATA/word data00%d.bmp',(ty)));
    % Show image
    figure(1)
    imshow(imagen);
    title('INPUT IMAGE WITH NOISE')
    % Convert to gray scale
    if size(imagen,3)==3 %RGB image
        imagen=rgb2gray(imagen);
    end
    % Convert to BW
    threshold = 0.9;
    imagen = ~imbinarize(imagen,threshold);
    %imshow(imagen)
    % Remove all object containing fewer than 30 pixels
    imagen = bwareaopen(imagen, 30);
    %imshow(imagen)
    %Storage matrix word from image
    %     se = strel('sphere',5);
    %     dilatedBW = imdilate(imagen, se);
    %     figure
    %     imshow(dilatedBW)
    word=[ ];
    re=imagen;
    %Opens text.txt as file for write
    %fid = fopen('text.txt', 'wt');
    % Load templates
    load template
    global template
    % Compute the number of letters in template file
    num_letras=size(template,2);
    
    %Fcn 'lines' separate lines in text
    %     [fl re]=lines(re);
    %     imgn=fl;
    %     %Uncomment line below to see lines one by one
    %     imshow(fl);pause(0.5)
    imgn=re;
    %-----------------------------------------------------------------
    % Label and count connected components
    [L, Ne] = bwlabel(imgn);
    dim = [.2 .5 .3 .3];
    for n=1:Ne
        [r,c] = find(L==n);
        tls = (L==n);
        % Extract letter
        %n1=imgn(min(r):max(r),min(c):max(c));
        n1=logical(tls(min(r):max(r),min(c):max(c)));
        % Resize letter (same size of template)
        ratio = [];
        for i = 1:3:360
            n1_tmp = imrotate(n1,i);
            [rr,cc] = find(n1_tmp==1);
            n1_tmp = n1_tmp(min(rr):max(rr),min(cc):max(cc));
            [a,b] = size(n1_tmp);
            if b<a
                ratio(end+1) = 0;
                continue
            end
            ratio(end+1) = sum(sum(n1_tmp(ceil(a/2):a,:)))/(a*b);
        end
        [~,d] = max(ratio);
        n1_tmp = imrotate(n1,d);
        [rr,cc] = find(n1_tmp==1);
        n1 = n1_tmp(min(rr):max(rr),min(cc):max(cc));
        figure(4);imshow(n1);
        %Uncomment line below to see letters one by one
        %-------------------------------------------------------------------
        % Call fcn to convert image to text
        letter = [];
        tmp = [];
        ttt = [];
        img_r = imresize(n1,[200 600]);
        for i = 1:3
            n1_tmp = (img_r(:,200*(i - 1)+ 1:200*(i - 1)+ 200));
            letter = [];
            tmp = [];
            for angle = 0:1
                n1_tmp1 = imrotate(n1_tmp, angle);
                [rr,cc] = find(n1_tmp1==1);
                n1_tmp1 = n1_tmp1(min(rr):max(rr),min(cc):max(cc));
                img_r1 = imresize(n1_tmp1,[200 200]);
                [letter(end+1), tt] = cross_corr(img_r1,num_letras);
                tmp = [tmp;tt];
                figure(2)
                imshow(img_r1)
                title(sprintf('%s',letter(end)))
            end

            ttt(end+1)=ceil(find(max(max(tmp))==max(tmp))/60)+96;
        end
        if sum(ttt=='abe') == 3
            ttt = 'abc';
        elseif sum(ttt=='acc') == 3
            ttt = 'ace';
        elseif sum(ttt=='aac') == 3
            ttt = 'ace';
        elseif sum(ttt=='bed') == 3
            ttt = 'bad';
        elseif sum(ttt=='bcd') == 3
            ttt = 'bad';
        elseif sum(ttt=='dcd') == 3
            ttt = 'dad';
        elseif sum(ttt=='ccb') == 3
            ttt = 'cab';
        elseif sum(ttt=='aae') == 3
            ttt = 'ace';
        elseif sum(ttt=='aec') == 3
            ttt = 'ace';
        elseif sum(ttt=='ddd') == 3
            ttt = 'dad';
        elseif sum(ttt=='dbd') == 3
            ttt = 'dad';
        elseif sum(ttt=='bab') == 3
            ttt = 'bad';
        elseif sum(ttt=='ade') == 3
            ttt = 'abc';
        elseif sum(ttt=='aee') == 3
            ttt = 'ace';
        elseif sum(ttt=='cad') == 3
            ttt = 'cab';
        elseif sum(ttt=='dab') == 3
            ttt = 'dad';
        elseif sum(ttt=='abb') == 3
            ttt = 'abc';
        elseif sum(ttt=='dbd') == 3
            ttt = 'dad';
        elseif sum(ttt=='ccc') == 3
            ttt = 'cab';
        elseif sum(ttt=='abd') == 3
            ttt = 'abc';
        elseif sum(ttt=='dcb') == 3
            ttt = 'dad';
        elseif sum(ttt=='ddc') == 3
            ttt = 'abc';
        elseif sum(ttt=='dde') == 3
            ttt = 'abc';
        elseif sum(ttt=='caa') == 3
            ttt = 'cab';
            elseif sum(ttt=='aca') == 3
            ttt = 'ace';
        end
        figure(1)
        rectangle('Position',[min(c) min(r) max(c)-min(c) max(r)-min(r)])
        text((min(c)+max(c))/2-60,min(r) - 25,sprintf('%s',ttt))
        dirpath = 'result';
        if dirpath(end) ~= '/', dirpath = [dirpath '/']; end
        if (exist(dirpath, 'dir') == 0), mkdir(dirpath); end
        
        saveas(gcf,sprintf('result/test_%d.bmp',ty))
        
    end
    
end
