AA = imread('grizzlypeakg.png');
[m1,n1] = size(AA);
t1_start = clock;


for k = 1:1000
    A = randi([0 255],[m1,n1]);
    for i=1:m1
        for j=1:n1
            if A(i,j) <= 10
                A(i,j) = 0;
            end
        end
    end
end

t1_end = clock;
disp(t1_end - t1_start);


t2_start = clock;

for k = 1:1000
    A = randi([0 255],[m1,n1]);
%     B = A<=10;
%     A(B) = 0;
end

t2_end = clock;
disp(t2_end - t2_start);