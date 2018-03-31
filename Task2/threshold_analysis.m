% threshold_analysis.m script
%   analyses the effect of the threshold on the classification accuracy

% Clear all variables and close figures
clear variables; close all;

% load the data set
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat');

% Feature vectors: Convert uint8 data to double (but do not divide by 255)
Xtrn = double(dataset.train.images);
Xtst = double(dataset.test.images);
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

% Run classification with different thresholds
thresholds = [1, 10:10:255];
for t = thresholds
    Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, t);
    [cm, acc] =  my_confusion(Ctst, Cpreds);
    fprintf('\nThreshold: %d, Accuracy: %.3f', t, acc);
end
fprintf('\n');
thresholds = 35:5:65;
for t = thresholds
    Cpreds = my_bnb_classify(Xtrn, Ctrn, Xtst, t);
    [cm, acc] =  my_confusion(Ctst, Cpreds);
    fprintf('\nThreshold: %d, Accuracy: %.3f', t, acc);
end
fprintf('\n');
