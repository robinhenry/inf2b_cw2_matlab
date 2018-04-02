function [mu] = myMean(matrix)
% Input:
%   matrix: L-by-D data matrix
% Output:
%   mu: D-by-1 column vector of sample mean values, where mu(i) = mean(matrix(:,i)).

% Check if the matrix is not empty to make sure we do not divide by 0.
if (size(matrix,1) == 0) 
    s = 1;
else 
    s = size(matrix,1);
end
% Compute sample mean vector
mu = ( sum(matrix, 1) ./ s )';
end