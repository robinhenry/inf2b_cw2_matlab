% test

% For each k in kb:
for i = 1:size(kb,2)
   % Get confusion matrix and accuracy
   [CM, acc] =  my_confusion(Ctst, Cpreds(:,i));
   % Save each confusion matrix.
   eval(sprintf('cm%d = CM;', kb(i)));
   s = sprintf('cm%d', kb(i));
   save(strcat(s, '.mat'), s);
   % Display the required information - k, N, Nerrs, acc.
   N = size(Ctst,1);
   Nerrs = N * (1-acc);
   fprintf('k: %d, N: %d, Nerrs: %d, acc: %d', kb(i), N, Nerrs, acc);
end