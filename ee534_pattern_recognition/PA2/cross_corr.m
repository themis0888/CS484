function [letter, tt] = cross_corr(imagn,num_letras)
global template
comp=[ ];
for n=1:num_letras
    sem = corr2((template{1,n}),(imagn));
    comp = [comp sem];
end
vd = find(comp==max(comp));
tmp = ceil(vd/60);
letter_list = 'abcde';

tt = comp;
letter = letter_list(tmp);
end

