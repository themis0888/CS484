%CREATE TEMPLATES
%Letter
templates1 = {};
for i = 1:60
    templates1(end + 1) = {imread(sprintf('data_split/a%d.bmp',i))};
end
for i = 1:60
    templates1(end + 1) = {imread(sprintf('data_split/b%d.bmp',i))};
end
for i = 1:60
    templates1(end + 1) = {imread(sprintf('data_split/c%d.bmp',i))};
end
for i = 1:60
    templates1(end + 1) = {imread(sprintf('data_split/d%d.bmp',i))};
end
for i = 1:60
    templates1(end + 1) = {imread(sprintf('data_split/e%d.bmp',i))};
end
save ('templates1','templates1')
clear all
