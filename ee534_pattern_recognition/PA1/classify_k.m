function ErrRate = classify_k(W1_train, W2_train, W3_train, W1_test, W2_test, W3_test, k)

tr_data = [W1_train; W2_train; W3_train];
te_data = [W1_test; W2_test; W3_test];

% k = 150;

% prediction list, returns the predicted class 
predict_list = [];
data_size = size(tr_data);
num_data = data_size(1);

for i = 1:num_data
    instance = te_data(i,:);
    % Euclidean distance
    euc_dist = sqrt(sum((tr_data - instance).^2,2));
    % Manhattan distance
    man_dist = sum(abs(tr_data - instance),2);
    dist = euc_dist;
    
    [~, ranking] = sort(dist, 'ascend');
    sum_W1 = sum(ranking(1:k)<=300);
    sum_W2 = sum(301<=ranking(1:k) & ranking(1:k) <= 600);
    sum_W3 = sum(601<=ranking(1:k));
    
    decision = [sum_W1, sum_W2, sum_W3];
    [~, predict] = max(decision);
    predict_list = [predict_list, predict]; 
end

Error_W1 = 300 - sum(predict_list(1:300)==1);
Error_W2 = 300 - sum(predict_list(301:600)==2);
Error_W3 = 300 - sum(predict_list(601:900)==3);
Error_total = Error_W1 + Error_W2 + Error_W3;

ErrRate = Error_total/num_data;

end

    
    