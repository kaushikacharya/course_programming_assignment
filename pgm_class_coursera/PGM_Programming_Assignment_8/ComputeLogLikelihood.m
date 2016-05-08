function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
for data_i = 1:N
    prob_class_array = NaN(1,K);
    for class_i = 1:K
        prior_class = P.c(class_i);
        if length(size(G)) == 2
            G_class = G;
        else
            G_class = G(:,:,class_i);
        end
        
        logProb = 0;
        for part_i = 1:size(G,1)
            % y,x,alpha
            if G_class(part_i,1) == 0
                % only class variable is the parent
                
                % eqn (2)
                logProb = logProb + lognormpdf(dataset(data_i,part_i,1), P.clg(part_i).mu_y(class_i), P.clg(part_i).sigma_y(class_i));
                
                % eqn (3)
                logProb = logProb + lognormpdf(dataset(data_i,part_i,2), P.clg(part_i).mu_x(class_i), P.clg(part_i).sigma_x(class_i));

                % eqn (4)
                logProb = logProb + lognormpdf(dataset(data_i,part_i,3), P.clg(part_i).mu_angle(class_i), P.clg(part_i).sigma_angle(class_i));
                
            elseif G_class(part_i,1) == 1
                % another body part is a parent of this body part alongwith
                % class variable
                parent_part_i = G_class(part_i,2);
                
                % eqn (5)
                mu_y = P.clg(part_i).theta(class_i,1:4) * [1 dataset(data_i,parent_part_i,1) dataset(data_i,parent_part_i,2) dataset(data_i,parent_part_i,3)]';
                logProb = logProb + lognormpdf(dataset(data_i,part_i,1), mu_y, P.clg(part_i).sigma_y(class_i));
                 
                % eqn (6)
                mu_x = P.clg(part_i).theta(class_i,5:8) * [1 dataset(data_i,parent_part_i,1) dataset(data_i,parent_part_i,2) dataset(data_i,parent_part_i,3)]';
                logProb = logProb + lognormpdf(dataset(data_i,part_i,2), mu_x, P.clg(part_i).sigma_x(class_i));

                % eqn (7)
                mu_angle = P.clg(part_i).theta(class_i,9:12) * [1 dataset(data_i,parent_part_i,1) dataset(data_i,parent_part_i,2) dataset(data_i,parent_part_i,3)]';
                logProb = logProb + lognormpdf(dataset(data_i,part_i,3), mu_angle, P.clg(part_i).sigma_angle(class_i));

            else
                fprintf(1,'unknown value');
            end
        end
        
        prob_class_array(class_i) = prior_class*exp(logProb);
        % logProb = logProb + log(prior_class);
        % loglikelihood = loglikelihood + logProb;
    end
    loglikelihood = loglikelihood + log(sum(prob_class_array));
end