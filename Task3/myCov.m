function Covs = myCov(matrix, mu)
% Input:
%   matrix : L-by-D data matrix
%   mu     : D-by-1 sample mean vector
% Output:
%   Covs: D-by-D sample covariance matrix.

% Size
L = size(matrix, 1);

% Compute sample covariance matrix
diff = matrix' - repmat(mu, 1, L);
Covs = (diff * diff') / L;

end