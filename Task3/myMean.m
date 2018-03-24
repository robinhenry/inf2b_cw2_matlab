function [mu] = myMean(matrix)
% Input:
%   matrix: L-by-D data matrix
% Output:
%   mu: D-by-1 column vector of sample mean values, where mu(i) = mean(:,i).

% Compute sample mean vector
mu = ( sum(matrix, 1) ./ size(matrix,1) )';
end