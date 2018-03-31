function [Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)
% Bayes classification with multivariate Gaussian distributions.
% Input:
%   Xtrn : M-by-D training data matrix
%   Ctrn : M-by-1 label vector for Xtrn
%   Xtst : N-by-D test data matrix
%   epsilon : A scalar parameter for regularisation
% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst
%  Ms     : D-by-K matrix of mean vectors
%  Covs   : D-by-D-by-K 3D array of covariance matrices

% Size of matrices
D = size(Xtrn,2);
N = size(Xtst,1);
K = 26;             % number of classes

% Compute matrix of sample mean vectors 
%   & 3D array of sample covariance matrices (including regularisation)
Ms = zeros(D,K);
Covs = zeros(D,D,K);
for k = 1:K
    samples = Xtrn(Ctrn == k, :);           % training samples from class k
    mu = myMean(samples);
    Ms(:,k) = mu;
    Covs(:,:,k) = myCov(samples, mu) + eye(D) * epsilon;
end

% NB: No need to include the prior probability to compute the posterior
%     probability, since we assume a uniform prior distribution over class

% Compute posterior probabilities for the test samples, in the log domain
post_log = zeros(N, K);
for k = 1:K
    mu = Ms(:,k);
    sigma = Covs(:,:,k);
    diff = Xtst' - repmat(mu, 1, N);
    post_matrix = - 0.5 * diff' * inv(sigma) * diff - 0.5 * logdet(sigma);
    post_log(:,k) =  diag(post_matrix);
end

% Choose the class corresponding to the max posterior probability, for each test sample
[~, Cpreds] = max(post_log, [], 2);