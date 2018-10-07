image1 = im2single(imread('../data/dog.bmp'));

tic
for i = 1:10
    im = rand(8000,2000);
    fi = rand(15,15);
    tic
    im2 = imfilter(im, fi);
    toc
end
toc