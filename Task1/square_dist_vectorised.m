function DI = square_dist_vectorised(Xtrn, Xtst)

M = size(Xtrn, 1);          % number of training samples
N = size(Xtst, 1);          % number of test samples

XX = sum(Xtst .^ 2, 2);
YY = sum(Xtrn .^ 2, 2);

DI = repmat(XX, 1, M) - 2 * Xtst * Xtrn' + repmat(YY, 1, N)'; 

end