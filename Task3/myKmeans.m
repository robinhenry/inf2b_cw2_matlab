function idx = myKmeans (data, K)
% Take a dataset A and random initial cluster centres, and apply 
%  the K-means algorithm. Returns the indexes of clusters and the errors after each iteration. 

S = size(data,1);              % number of samples
D = size(data,2);             % number of elements in each sample
maxiter = 500;              % Maximum number of iterations
s = rng(1);                 % set random seed
centres = rand(K,D);

Dist = zeros(K, S);         % K-by-S matrix for storing distances
                            %   between cluster centres and observations
                            
idx_prev = zeros(1, S);     % 1-by-S vector for storing previous assignment
                            
%fprintf('[0] Iteration: \n');

% Iterate 'maxiter' times
for i = 1:maxiter
    % Compute Squared Euclidean distance (i.e. the squared distance)
    %   between each cluster centre and each observation
    for c = 1:K
        Dist(c,:) = square_dist(data, centres(c,:));
    end
    
    % Assign data to clusters
    % idx are the cluster assignments
    [~, idx] = min(Dist, [], 1);     % find min dist. for each observation
    
    
    % Check if the assignments have changed
    if (idx == idx_prev)
       % fprintf('K-means converged after iteration [%d]\n', i-1);
        break
    end
    
    % Update cluster centres
    for c = 1:K
        % check the number of samples assigned to this cluster
        if (sum(idx==c) == 0)
          %  fprintf('cluser %d is empty', c);
        else 
            centres(c, :) = mean( data(idx==c,:) );
        end
    end
    
   % fprintf('[%d] Iteration: \n', i)
    
    % Store current assignments
    idx_prev = idx;
end
end