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
for i = 1:K
    
   % Vector of predictions corresponding to truth 'i'
   preds = Cpreds(Ctrues == i);
   
   % Increment the ith row in CM 
   for j = preds'
       CM(i,j) = CM(i,j) + 1;
   end 
end

% Compute accuracy
acc = trace(CM) / size(Ctrues, 1);

end