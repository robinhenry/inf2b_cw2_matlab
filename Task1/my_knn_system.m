% Script my_knn_system.m

% Clear all variables and close figures
clear variables; close all;


% load the data set
%load("/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat");
load('../data/data.mat');

% Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

% Prepare measuring time
tic;
% Run K-NN classification
kb = [1,3,5,10,20];
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb);

% Measure the time taken, and display it.
elapsed_time = toc;
disp(elapsed_time);
% Get confusion matrix and accuracy for each k in kb.
for i = 1:size(kb,2)
   [CM, acc] =  my_confusion(Ctst, Cpreds(:,i));
   eval(sprintf('cm%d = CM', kb(i)));
   s = sprintf('cm%d', kb(i));
   save(strcat(s, '.mat'), s);
end
% Save each confusion matrix.

% Display the required information - k, N, Nerrs, acc for
%           each element of kb.



