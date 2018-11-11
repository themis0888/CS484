function ErrRate = classify(W1_train, W2_train, W3_train, W1_test, W2_test, W3_test)

% Variable K
K = 151;
train_data = [W1_train; W2_train; W3_train];
test_data = [W1_test; W2_test; W3_test];
test_result = zeros(900,1);

for i = 1:900
    temp_test = test_data(i,:);
    % Calculate euclidean distance and sort them
    dist = sum((train_data - temp_test).^2,2);
    [sorted_dist, index] = sort(dist,1);
    index = index(1:K);
    num_W1 = sum(index<=300);
    num_W2 = sum(300<index & index<=600);
    num_W3 = sum(600<index & index<=900);
    
    boundary = [num_W1, num_W2, num_W3];
    [pred_num, pred] = max(boundary);
    
    if sum(boundary==pred_num) ~= 1
        dist_W1 = sum(dist(index(index<=300)));
        dist_W2 = sum(dist(index(300<index & index<=600)));
        dist_W3 = sum(dist(index(600<index & index<=900)));
        dist = [dist_W1, dist_W2, dist_W3];
        [no_use, pred] = min(dist);
        test_result(i) = pred;
    else
        test_result(i) = pred;
    end
end

wrong_W1 = sum(test_result(1:300)~=1);
wrong_W2 = sum(test_result(301:600)~=2);
wrong_W3 = sum(test_result(601:900)~=3);
wrong = wrong_W1 + wrong_W2 + wrong_W3;

ErrRate = wrong/900;

     
end

