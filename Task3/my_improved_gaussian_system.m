#
# Template for my_improved_gaussian_system.m
#
# load the data set
#   NB: replace <UUN> with your actual UUN.
load("/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/<UUN>/data.mat");

# Feature vectors: Convert uint8 data to double, and divide by 255.
Xtrn = double(dataset.train.images) ./ 255.0;
Xtst = double(dataset.test.images) ./ 255.0;
# Labels
Ctrn = dataset.train.labels;
Ctst = dataset.test.labels;

#YourCode - Prepare to measure time

# Run classification
# epsilon = 0.01;
# [Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon);
[Cpreds] = my_improved_gaussian_classify(Xtrn, Ctrn, Xtst);

#YourCode - Measure the time taken, and display it.

#YourCode - Get a confusion matrix and accuracy

#YourCode - Save the confusion matrix as "Task3/cm_improved.mat".

#YourCode - Display information if any





  
