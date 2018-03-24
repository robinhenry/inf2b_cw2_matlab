function [CM, acc] = my_confusion(Ctrues, Cpreds)
% Input:
%   Ctrues : N-by-1 ground truth label vector
%   Cpreds : N-by-1 predicted label vector
% Output:
%   CM : K-by-K confusion matrix, where CM(i,j) is the number of samples whose target is the ith class that was classified as j
%   acc : accuracy (i.e. correct classification rate)

% Initialisation of confusion matrix
K = 26;
CM = zeros(K, K);

% Iterate over each class
for k = 1:K
   % Compute vector of predictions corresponding to truth of class k
   preds = Cpreds(Ctrues == k);
   % Increment the kth row (samples that should be of class k) in CM 
   for j = preds'
       CM(k,j) = CM(k,j) + 1;
   end 
end

% Compute accuracy
acc = trace(CM) / size(Ctrues, 1);

end