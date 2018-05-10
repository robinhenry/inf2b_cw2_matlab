function [Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
% Classification using Naive Bayes with multivariate Bernoulli distributions
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   threshold : A scalar parameter for binarisation
% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst

% Matrix sizes
N = size(Xtst, 1);          % number of test samples
D = size(Xtrn, 2);          % size of a feature vector
C = 26;                     % number of classes

% Binarisation of Xtrn and Xtst
Xtrn_bin = Xtrn >= threshold;
Xtst_bin = Xtst >= threshold;

% Matrices to store characteristics of each class
trn_zeros = zeros(C, D);                % C-by-D matrix, numbers of 0's for each pixel
trn_ones = zeros(C, D);                 % C-by_D matrix, numbers of 1's for each pixel

% Initialise matrix to store likelihoods
likelihoods = zeros(N, C);

for k = 1:C
    % Select training samples from class k and add 1 row in case there is
    %   only 1 training sample (just in case).
    Xtrn_k = [Xtrn_bin(Ctrn == k, :); zeros(1,D)];
    % Number of samples of class k
    nbr_samples_k = size(Xtrn_k, 1) - 1;      % '-1' because a row was added before
    % P(D_i = 0|C_k), where D_i is the ith element of a feature vector D
    %   + using Laplace's rule to avoid 0's
    trn_zeros(k,:) = (sum(Xtrn_k == 0) -1) / (nbr_samples_k);
    % P(D_i = 1|C_k), where D_i is the ith element of a feature vector D
    %   + using Laplace's rule to avoid 0's
    trn_ones(k,:) = (sum(Xtrn_k == 1)) / (nbr_samples_k);
    
    % Compute the likelihoods
    factor_0 = repmat(trn_zeros(k,:),N,1) .^ (1-Xtst_bin);
    factor_1 = repmat(trn_ones(k,:),N,1) .^ Xtst_bin;
    
    likelihoods(:,k) = prod(factor_0 .* factor_1, 2);
end

% NB: No need to multiply the likelihoods by the prior probability, 
%     since we assume a uniform prior distribution over class

% Get the maximum posterior probability and find Cpreds
[~, Cpreds] = max(likelihoods, [], 2);

end
