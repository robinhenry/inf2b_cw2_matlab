function [Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   threshold : A scalar parameter for binarisation
% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst

% Matrix sizes
M = size(Xtrn, 1);          % number of training samples
N = size(Xtst, 1);          % number of test samples
D = size(Xtrn, 2);          % size of a feature vector
C = 26;                     % number of classes

% Binarisation of Xtrn and Xtst.
Xtrn_bin = Xtrn >= threshold;
Xtst_bin = Xtst >= threshold;

% Naive Bayes classification with multivariate Bernoulli distributions

% Matrices to store characteristics of each class.
trn_zeros = zeros(C, D);                % C-by-D matrix, numbers of 0's for each pixel
trn_ones = zeros(C, D);                 % C-by_D matrix, numbers of 1's for each pixel
prior = zeros(1, C);                    % 1-by-C vector, prior probabilitiy for each class

for k = 1:C
    % Select training samples from class k and add 2 rows in case there is only 0 or 1
    Xtrn_k = [Xtrn_bin(Ctrn == k, :); zeros(2,D)];
    % Prior probability for eah class k (good to compute even if we know it is 1/26)
    nbr_samples_C = size(Xtrn_k, 1) - 2;      % '-2' because 2 rows were added before
    prior(k) = nbr_samples_C / M;
    % P(D_i = 0|C_k) -> using Laplace's rule to avoid 0's
    trn_zeros(k,:) = (sum(Xtrn_k == 0) -2 + 1) / (nbr_samples_C + C);
    % P(D_i = 1|C_k) -> using Laplace's rule to avoid 0's
    trn_ones(k,:) = (sum(Xtrn_k == 1) + 1) / (nbr_samples_C + C);
end

% Compute the likelihood matrix (N-by-C)
likelihoods = zeros(N, C);
for k = 1:C
    bi_0 = trn_zeros(k,:) .^ (1-Xtst_bin);
    bi_1 = trn_ones(k,:) .^ Xtst_bin;
    likelihoods(:,k) = prod(bi_0 .* bi_1, 2);
end

% Compute posterior probability matrix (N-by-C)
post = likelihoods .* repmat(prior, N, 1);

% Get the maximum posterior probability and find Cpreds
[max_prob, Cpreds] = max(post, [], 2);

end
