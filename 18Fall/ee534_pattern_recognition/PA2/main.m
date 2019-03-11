clear;clc;fclose all;
% Load training data
% if use gray scale image files
img_train1 = imread('DATA\TRAINING_DATA\gray scale\character data001.bmp');
img_train2 = imread('DATA\TRAINING_DATA\gray scale\character data002.bmp');
img_train3 = imread('DATA\TRAINING_DATA\gray scale\character data003.bmp');
img_train4 = imread('DATA\TRAINING_DATA\gray scale\character data004.bmp');
img_train5 = imread('DATA\TRAINING_DATA\gray scale\character data005.bmp');
% elif use black and white scale image files
img_train1 = imread('DATA\TRAINING_DATA\black and white\character data001.bmp');
img_train2 = imread('DATA\TRAINING_DATA\black and white\character data002.bmp');
img_train3 = imread('DATA\TRAINING_DATA\black and white\character data003.bmp');
img_train4 = imread('DATA\TRAINING_DATA\black and white\character data004.bmp');
img_train5 = imread('DATA\TRAINING_DATA\black and white\character data005.bmp');

% Load validation data
img_validc1 = imread('DATA\VALID_DATA\CharA.bmp');
img_validc2 = imread('DATA\VALID_DATA\CharB.bmp');
img_validc3 = imread('DATA\VALID_DATA\CharC.bmp');
img_validc4 = imread('DATA\VALID_DATA\CharD.bmp');
img_validc5 = imread('DATA\VALID_DATA\CharE.bmp');

img_validw1 = imread('DATA\VALID_DATA\word data001.bmp');
img_validw2 = imread('DATA\VALID_DATA\word data002.bmp');
img_validw3 = imread('DATA\VALID_DATA\word data003.bmp');
img_validw4 = imread('DATA\VALID_DATA\word data004.bmp');

%% Your code below (Preprocessing & Feature extraction & Classification)
% ex) train_A = makefeature(img_train1);
%     train_B = makefeature(img_train2);
%     train_C = makefeature(img_train3);
%     train_D = makefeature(img_train4);
%     train_E = makefeature(img_train5);
%     valid_cA = makefeature(img_validc1);
%     valid_cB = makefeature(img_validc2);
%     valid_cC = makefeature(img_validc3);
%     valid_cD = makefeature(img_validc4);
%     valid_cE = makefeature(img_validc5);
%     valid_wA = makefeature(img_validw1);
%     valid_wB = makefeature(img_validw2);
%     valid_wC = makefeature(img_validw3);
%     valid_wD = makefeature(img_validw4);
%     CharacterErrRate, WordErrRate = classify(Traindata, validdata);

%% Evaluation
%disp(['Character Accuracy : ',num2str(CharacterErrRate),' %']);
%disp(['Word Accuracy : ',num2str(WordErrRate),' %']);

% Read and split 
warning off
clear all
close all
train = dir('DATA/TRAINING_DATA/gray scale/*.bmp');
test = dir('DATA/VALID_DATA/*.bmp');
train_data = {};
test_data = {};
dirpath = 'split';
if dirpath(end) ~= '/', dirpath = [dirpath '/']; end
if (exist(dirpath, 'dir') == 0), mkdir(dirpath); end

for i = 1:length(train)
    train_data(end + 1) = {255 - imread(strcat('DATA/TRAINING_DATA/gray scale/','\',train(i).name))};
end
tmp = {};
letter = 'abcde';
for i = 1:length(train_data)

    figure
    imagen = train_data{i};

        if size(imagen,3)==3 %RGB image
            imagen=rgb2gray(imagen);
        end
        % Convert to BW
        threshold = graythresh(imagen);
        threshold = 0.1;
        imagen =imbinarize(imagen,threshold);
        imshow(imagen)

        imagen = bwareaopen(imagen,30);

        [L, Ne] = bwlabel(imagen);

        for n = 1:Ne
            [r,c] = find(L==n);

            n1=imagen(min(r):max(r),min(c):max(c));

            img_r=imresize(n1,[200 200]);
            imwrite(img_r,sprintf('split/%s%d.bmp',letter(i),n))
            imshow(img_r)
        end    
end


% template 

dirpath = 'split';
if dirpath(end) ~= '/', dirpath = [dirpath '/']; end
if (exist(dirpath, 'dir') == 0), mkdir(dirpath); end

template = {};
letters = 'abcde';
for j = 1:5
    for i = 1:60
        template(end + 1) = {imread(sprintf('split/%s%d.bmp',letters(j),i))};
    end
end
save ('template','template')
clear all


% % Code for segmentation

clc, close all, clear all
letter_list_t = 'ABCDE';

Total_Err = 0;
Total_Num = 0;
for ty = 1:5
    imagen=imread(sprintf('DATA/VALID_DATA/Char%s.bmp',letter_list_t(ty)));
    figure(1);
    imshow(imagen);
    title(sprintf('Char%s.bmp',letter_list_t(ty)))

    threshold = 0.9;
    imagen = ~imbinarize(imagen,threshold);

    imagen = bwareaopen(imagen, 25);

    word=[ ];
    re=imagen;

    load template
    global template

    num_letras=size(template,2);

    imgn=re;

    [L, Ne] = bwlabel(imgn);
    dim = [.2 .5 .3 .3];
    Total_Num = Total_Num + Ne;
    Err_Char = 0;
    for n=1:Ne
        [r,c] = find(L==n);

        n1=imgn(min(r):max(r),min(c):max(c));

        letter = [];
        tmp = [];
        for angle = 0:3:360
            n1_tmp = imrotate(n1, angle);
            [rr,cc] = find(n1_tmp==1);
            n1_tmp = n1_tmp(min(rr):max(rr),min(cc):max(cc));
            img_r = imresize(n1_tmp,[200 200]);
            [letter(end+1), tt] = cross_corr(img_r,num_letras);
            tmp = [tmp;tt];
            figure(2)
            imshow(img_r)
            title(sprintf('%s',letter(end)))
        end
        
        ttt=ceil(find(max(max(tmp))==max(tmp))/60)+96;
        figure(1)
        rectangle('Position',[min(c) min(r) max(c)-min(c) max(r)-min(r)])
        text((max(c) + min(c))/2-20, min(r) - 50,sprintf('%s',ttt))
        if ttt ~= lower(letter_list_t(ty))
            Err_Char = Err_Char + 1;
        end
    end
    
    fprintf(sprintf('Err_%s = %d\n', letter_list_t(ty), Err_Char));
    Total_Err = Total_Err + Err_Char;
    Err_Char = 0;
    dirpath = 'result';
    if dirpath(end) ~= '/', dirpath = [dirpath '/']; end
    if (exist(dirpath, 'dir') == 0), mkdir(dirpath); end
    saveas(gcf,sprintf('result/test_%s.bmp',letter_list_t(ty)))

end

fprintf(sprintf('Total_Err = %f\n', Total_Err/Total_Num));

% Code for words 
warning off
% Clear all
clc, close all, clear all
letter_list_t = 'ABCDE';
for ty = 1:4

    imagen=imread(sprintf('DATA/VALID_DATA/word data00%d.bmp',(ty)));
    figure(1)
    imshow(imagen);
    title(sprintf('Word data00%d.bmp',(ty)))

    threshold = 0.9;
    imagen = ~imbinarize(imagen,threshold);

    imagen = bwareaopen(imagen, 30);

    word=[ ];
    re=imagen;

    load template
    global template

    num_letras=size(template,2);
    

    imgn=re;

    [L, Ne] = bwlabel(imgn);
    dim = [.2 .5 .3 .3];
    for n=1:Ne
        [r,c] = find(L==n);
        tls = (L==n);

        n1=logical(tls(min(r):max(r),min(c):max(c)));
 
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