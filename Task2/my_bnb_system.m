% my_bnb_system.m script

% Clear all variables and close figures
clear variables; close all;

% load the data set
%load("/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat");
load("../data/data.mat");

% Feature vectors: Convert uint8 data to double (but do not divide by 255)
Xtrn = double(dataset.train.images);
Xtst = double(dataset.test.images);
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

% Prepare to measure time
bnb_start = tic;

% Run classification
threshold = 1;
Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold);

% Measure the time taken, and display it.
bnb_end = toc(bnb_start);
fprintf("Time taken by my_bnb_classify(): %d seconds\n", bnb_end);

% Get a confusion matrix and accuracy
[cm, acc] =  my_confusion(Ctst, Cpreds);

% Save the confusion matrix as "Task2/cm.mat".
save("cm.mat", "cm");

% Display the required information - N, Nerrs, acc.
N = size(Ctst,1);
Nerrs = N * (1-acc);
fprintf('N: %d, Nerrs: %d, acc: %d\n', N, Nerrs, acc);
