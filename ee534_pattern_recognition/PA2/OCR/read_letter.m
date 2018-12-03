function [letter, tt] = read_letter(imagn,num_letras)
% Computes the correlation between template and input image
% and its output is a string containing the letter.
% Size of 'imagn' must be 42 x 24 pixels
% Example:
% imagn=imread('D.bmp');
% letter=read_letter(imagn)
global templates1
comp=[ ];
for n=1:num_letras
    sem = corr2((templates1{1,n}),(imagn));
    comp = [comp sem];
end
vd = find(comp==max(comp));
tmp = ceil(vd/60);
letter_list = 'abcde';
%*-*-*-*-*-*-*-*-*-*-*-*-*-
tt = comp;
letter = letter_list(tmp);
end

