% my_gaussian_system

% Clear all variables and close figures
clear variables; close all;

% load the data set
%load("/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat");
load("../data/data.mat");

% Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

% Prepare to measure time
gaussian_start = tic;

% Run classification
epsilon = 0.01;
[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon);

% Measure the time taken, and display it.
gaussian_end = toc(gaussian_start);
fprintf("Time taken by my_gaussian_classify: %d sec.\n", gaussian_end);

% Get a confusion matrix and accuracy
[cm, acc] =  my_confusion(Ctst, Cpreds);

% Save the confusion matrix as "Task3/cm.mat".
save("cm.mat", "cm");

% Save the mean vector and covariance matrix for class 26,
mu26 = Ms(:,26);
cov26 = Covs(:,:,26);
save("m26.mat", "mu26");
save("cov26.mat", "cov26");

%YourCode - Display the required information - N, Nerrs, acc.
N = size(Ctst,1);
Nerrs = N * (1-acc);
fprintf('N: %d, Nerrs: %d, acc: %d\n', N, Nerrs, acc);
