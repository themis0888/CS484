%CREATE TEMPLATES
%Letter

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
