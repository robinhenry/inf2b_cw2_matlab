function [sum_error] = sum_sq_error(A, centres, idx)
% Takes a data matrix A of points (x,y), a matrix of centres k x n and a
% vector indicating the classification of the points in the clusters.
% Returns the mean squared error.

sum_error = 0;

for k = 1:size(centres,1)
    
    k_points = A(idx == k, :);                                      % points in kluster k
    sq_diff = sum(bsxfun(@minus, k_points, centres(k,:)) .^ 2, 2); 
    sum_error = sum_error + (1/size(A,1)) * sum(sq_diff);
    
end
end

