function sq_dist = square_dist( U, v )
% Compute 1 x M row vector of square distances for M x N and 1 x N data U
%  and v, respectively.
    sq_dist = sum(bsxfun(@minus, U, v) .^2, 2)';
end