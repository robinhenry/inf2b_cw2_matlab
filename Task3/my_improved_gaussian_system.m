% my_improved_gaussian_system.m
%
% load the data set
load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1605269/data.mat');
%load('../data/data.mat');

% Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;
% Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

%YourCode - Prepare to measure time
gaussian_start = tic;

% Run classification
epsilon = 0.01;
for k=1:10
    [Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon,k);


    % Measure the time taken, and display it.
    gaussian_end = toc(gaussian_start);
    fprintf('\nTime taken by my_improved_gaussian_classify: %.3f seconds.\n', gaussian_end);

    %YourCode - Get a confusion matrix and accuracy
    [cm, acc] = my_confusion(Ctst, Cpreds);

    % Save the confusion matrix as "Task3/cm_improved.mat".
    save('cm_improved.mat', 'cm');

    % Display information
    N = size(Ctst,1);
    Nerrs = N * (1-acc);
    fprintf('k: %d, Num. of test samples: %d, Num. of errors: %4.f, Accuracy: %.3f.\n', k, N, Nerrs, acc);
end
