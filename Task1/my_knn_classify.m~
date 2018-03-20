function [Cpreds] = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   Ks   : L-by-1 vector of the numbers of nearest neighbours in Xtrn
% Output:
%  Cpreds : N-by-L matrix of predicted labels for Xtst

% Matrix sizes
M = size(Xtrn, 1);          % number of training samples
N = size(Xtst, 1);          % number of test samples
L = size(Ks, 2);            % number of different k-values to use

% Initialise return matrix
Cpreds = zeros(N, L);

% Compute distances between each test sample and each training sample
DI = square_dist_vectorised(Xtrn, Xtst);

% Sort the distances between each test sample and all the training samples
[dist_sorted, idx] = sort(DI, 2, 'ascend');          % idx = N-by-M matrix

% Iterate over each value of k from Ks
for i = 1:L   
    % Select the indexes corresponding to k nearest neighbours
    k = Ks(i);
    % Add 1 column in case k==1
    k_idx = [idx(:, 1:k) ones(N,1)];                      % k_idx = N-by-(k+1) matrix

    % Choose the most frequent class out of the k neighbours
    classes = Ctrn(k_idx);
    classes = classes(:,1:end-1);                   % remove last column
    Cpreds(:,i) =  mode(classes, 2);
end

end