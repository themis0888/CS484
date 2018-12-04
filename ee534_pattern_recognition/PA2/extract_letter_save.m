clear all
close all
train = dir('DATA/TRAINING_DATA/gray scale/*.bmp');
test = dir('DATA/TEST_DATA/*.bmp');
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
%     [a,b] = size(train_data{i});
%     aa = floor(a/10);
%     bb = floor(b/6);
    figure
    imagen = train_data{i};
%     l = 0;
%     for j = 0:9
%         for k = 0:5
%             l = l+1;
%             imagen = train_data{i}(aa*j + 1:aa*j+aa,bb*k+ 1:bb*k+bb);
            %subplot(10,6,l);imshow(imagen);
            % Convert to gray scale
            if size(imagen,3)==3 %RGB image
                imagen=rgb2gray(imagen);
            end
            % Convert to BW
            threshold = graythresh(imagen);
            threshold = 0.1;
            imagen =imbinarize(imagen,threshold);
            imshow(imagen)
            % Remove all object containing fewer than 30 pixels
            imagen = bwareaopen(imagen,30);
            % Label and count connected components
            [L, Ne] = bwlabel(imagen);
            %             if Ne>1
            %                 assert
            %             end
            for n = 1:Ne
                [r,c] = find(L==n);
                % Extract letter
                n1=imagen(min(r):max(r),min(c):max(c));
                % Resize letter (same size of template)
                img_r=imresize(n1,[200 200]);
                imwrite(img_r,sprintf('split/%s%d.bmp',letter(i),n))
                imshow(img_r)
            end
%         end
%     end
%     
    %imshow(train_data{i})
    
end