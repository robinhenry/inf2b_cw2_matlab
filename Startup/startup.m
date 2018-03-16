% Startup script

% Clear all variables and close figures
clear variables; close all;

% Load data
% load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat');
load('../data.mat');

% Training data
training_images = double(dataset.train.images);         % convert to double
training_labels = dataset.train.labels;

% Testing data
test_images = double(dataset.test.images);              % convert to double
test_labels = dataset.test.labels;