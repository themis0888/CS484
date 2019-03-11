function ErrRate = classify(W1_train, W2_train, W3_train, W1_test, W2_test, W3_test)

tr_data = [W1_train; W2_train; W3_train];
te_data = [W1_test; W2_test; W3_test];

% K-value
k = 150;

% prediction list, returns the predicted class 
predict_list = [];
data_size = size(tr_data);
num_data = data_size(1);

for i = 1:num_data
    instance = te_data(i,:);
    % Euclidean distance
    diff = abs(tr_data - instance);
    euc_dist = sqrt(sum(diff.^2,2));
    % Manhattan distance
    man_dist = sum(diff,2);
    dist = euc_dist;
    
    % ranking of the close datapoints
    [~, ranking] = sort(dist, 'ascend');
    num_class = [0, 0, 0];
    for j = 1:k
        if ranking(j)<=300
            num_class(1) = num_class(1) + 1;
        elseif ranking(j) <= 600
            num_class(2) = num_class(2) + 1;
        else
            num_class(3) = num_class(3) + 1;
        end
    end
    
    % prediction based on KNN 
    [~, predict] = max(num_class);
    predict_list = [predict_list, predict]; 
end

Error_W1 = 300 - sum(predict_list(1:300)==1);
Error_W2 = 300 - sum(predict_list(301:600)==2);
Error_W3 = 300 - sum(predict_list(601:900)==3);
Error_total = Error_W1 + Error_W2 + Error_W3;

ErrRate = Error_total/num_data;
end

    
    