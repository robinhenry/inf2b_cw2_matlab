function [Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon, K_clust)
% Input:
%   Xtrn    : M-by-D training data matrix
%   Ctrn    : M-by-1 label vector for Xtrn
%   Xtst    : N-by-D test data matrix
%   K-clust : number of clusters to create from each class
%  NB: you may add arguments if necessary
% Output:
%  Cpreds : N-by-1 matrix of predicted labels for Xtst

% Size of matrices
D = size(Xtrn,2);
M = size(Xtrn,1);
N = size(Xtst,1);
K = 26;             % number of classes

% New classification of Xtrn with more classes (clusters), using Kmeans algorithm
Ks = 1:(26 * K_clust);
new_Ctrn = zeros(M,1);

% K-means (for each class)
for k=1:K
    selection = Ctrn == k;                  % boolean vector to select S samples from class k
    samples = Xtrn(Ctrn == k, :);           % training samples S from class k
    idx = myKmeans(samples, K_clust)';      % S-by-1 vector of indexes corresponding to clusters
    new_Ctrn(selection) = idx + K_clust * (k-1);  
end

% Remove empty clusters
for k = Ks
    if(sum(new_Ctrn == k) == 0)
        Ks(Ks==k) = [];
    end
end

% Number of clusters after removing empty ones
num_clusters = size(Ks,2);

% Compute matrix of sample mean vectors 
%   & 3D array of sample covariance matrices (including regularisation)
Ms = zeros(D,num_clusters);
Covs = zeros(D,D,num_clusters);
for k = Ks
    samples = Xtrn(new_Ctrn == k, :);           % training samples from class k
    mu = myMean(samples);
    Ms(:,k) = mu;
    Covs(:,:,k) = myCov(samples, mu) + eye(D) * epsilon;
end

% NB: No need to include the prior probability to compute the posterior
%     probability, since we assume a uniform prior distribution over class

% Compute posterior probabilities for the test samples, in the log domain
post_log = zeros(N, num_clusters);
for k = Ks
    mu = Ms(:,k);
    sigma = Covs(:,:,k);
    diff = Xtst' - repmat(mu, 1, N);
    post_matrix = - 0.5 * diff' * inv(sigma) * diff - 0.5 * logdet(sigma);
    post_log(:,k) =  diag(post_matrix);
end

% Choose the class corresponding to the max posterior probability, for each test sample
[~, Cpreds] = max(post_log, [], 2);

% Go back to K classes, from num_clusters clusters
Cpreds = ceil(Cpreds/K_clust);

end