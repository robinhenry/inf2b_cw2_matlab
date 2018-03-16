function [CM, acc] = my_confusion(Ctrues, Cpreds)
% Creates a confusion matrix.

% Inputs:
%   Ctrues: N-by-1 vector of ground truth (target) class labels.
%   Cpreds: N-by-1 vector of predicted class labels.

% Outputs:
%   CM: K-by-K confusion matrix, where CM(i,j) is the number of samples whose target is 
%       the i?th class that was classified as j.  
%       K is the number of classes.
%   acc: A scalar variable representing the accuracy in the range [0,1].

% Initialisation of confusion matrix
K = 2;
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