% Script my_knn_system.m

% Clear all variables and close figures
clear variables; close all;

% Load data
% load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat');
load('../d/data.mat');
% Training data 
training_images = double(dataset.train.images) / 255.0;         % convert to double and within [0,1]
training_labels = dataset.train.labels;
% Testing data
test_images = double(dataset.test.images) / 255.0;              % convert to double and within [0,1]
test_labels = dataset.test.labels;

% Classification experiment
Ks = [1, 3, 5, 10, 20];                 % different k-values to use
tic;                                    % start the timer
predictions = my_knn_classify(training_images, training_labels, test_images, Ks);
elapsed_time = toc;


