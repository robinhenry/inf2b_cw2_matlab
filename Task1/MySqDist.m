function DI = MySqDist(Xtrn, Xtst)
% Compute square distances between 2 matrix of samples
% Inputs:
%   Xtrn: M-by-D matrix of M samples, each of dimension D
%   Xtst: N-by-D matrix of N samples, each of dimension D
% Ouptut:
%   DI: N-by-M euclidean square-distance matrix, where DI(i,j) is the distance
%       between sample Xtst(i,:) and sample Xtrn(j,:)

M = size(Xtrn, 1);          % number of training samples
N = size(Xtst, 1);          % number of test samples

% Compute the squared distance, using vectorisation
XX = sum(Xtst .^ 2, 2);
YY = sum(Xtrn .^ 2, 2);
DI = repmat(XX, 1, M) - 2 * Xtst * Xtrn' + repmat(YY, 1, N)'; 

end