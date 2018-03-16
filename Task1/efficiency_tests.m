% Script to decide what approach is the most efficient to use in
% my_knn_classify function

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

% Use fewer samples
M = 5000;
N = 1000;
training_images = training_images(1:M,:);
test_images = test_images(1:N,:);

% Experiments



