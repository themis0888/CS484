AA = imread('grizzlypeak.jpg');
[l1,m1,n1] = size(AA);
t1_start = clock;

for iter = 1:100
    A = randi([0 255], l1,m1,n1);
    for i=1:l1
        for j=1:m1
            for k=n1
                if A(i,j,k) <= 10
                    A(i,j,k) = 0;
                end
            end
        end
    end
end

t1_end = clock;
disp(t1_end - t1_start);


t2_start = clock;

for iter = 1:100
    A = randi([0 255], l1,m1,n1);
    B = A<=10;
    A(B) = 0;
end

t2_end = clock;
disp(t2_end - t2_start);