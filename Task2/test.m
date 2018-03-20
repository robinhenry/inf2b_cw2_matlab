% tests

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

% Only use a part of the data
train_samples = Xtrn(1:10,:);
test_samples = Xtst(1:3,:);
train_classes = Ctrn(1:10);

% Run classification
threshold = 1;
Cpreds = my_bnb_classify(train_samples, train_classes, test_samples, threshold);

