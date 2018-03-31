% Script my_knn_system.m

% Clear all variables and close figures
clear variables; close all;

% load the data set
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat');

% Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

% Prepare measuring time
knn = tic; 

% Run K-NN classification 
kb = [1,3,5,10,20];
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, kb);

% Measure the time taken, and display it
elapsed_time = toc(knn);
fprintf('\nTime taken by my_knn_classify(): %.3f seconds.\n\n', elapsed_time);

% For each k in kb:
for i = 1:size(kb,2)
   % Get confusion matrix and accuracy
   [CM, acc] =  my_confusion(Ctst, Cpreds(:,i));
   % Save each confusion matrix
   eval(sprintf('cm%d = CM;', kb(i)));
   s = sprintf('cm%d', kb(i));
   save(strcat(s, '.mat'), s);
   % Display the required information - k, N, Nerrs, acc
   N = size(Ctst,1);
   Nerrs = N * (1-acc);
   fprintf('k: %d,\t Num. of test samples: %i,\t Num. of errors: %4.f,\t Accuracy: %.3f.\n', kb(i), N, Nerrs, acc);
end
fprintf('\n');