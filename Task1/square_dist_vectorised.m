function DI = square_dist_vectorised(Xtrn, Xtst)

M = size(Xtrn, 1);          % number of training samples
N = size(Xtst, 1);          % number of test samples

XX = sum(Xtrn .^ 2, 2);
YY = sum(Xtst .^ 2, 2);

DI = repmat(XX, 1, N) - 2 * Xtrn * Xtst' + repmat(YY, 1, M)'; 

end