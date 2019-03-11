result = [];
k_list = 1:300;
for i = k_list
    Err = classify_k(W1_train, W2_train, W3_train, W1_test, W2_test, W3_test, i);
    result = [result; [i, Err]];
end

fig = plot(result(:,1), result(:,2));
title('Euclidean distance');
xlabel('K value');
ylabel('Error rate');

saveas(fig, 'Euclidean_distance.png');
[val, ind] = min(result(:,2));