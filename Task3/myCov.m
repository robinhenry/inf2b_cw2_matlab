function Covs = myCov(matrix, mu)
% Input:
%   matrix : L-by-D data matrix
%   mu     : D-by-1 sample mean vector
% Output:
%   Covs: D-by-D sample covariance matrix.

% Sizes
L = size(matrix, 1);
D = size(matrix, 2);

% Compute sample covariance matrix
diff = matrix' - repmat(mu, 1, L);
Covs = (diff * diff') / L;

end