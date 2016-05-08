function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% class probabilities
for i = 1:K
    P.c(i) = sum(labels(:,i))/size(labels,1);
end
clg_t = struct('mu_y',[],'sigma_y',[],'mu_x',[],'sigma_x',[],'mu_angle',[],'sigma_angle',[],'theta',[]);
P.clg(size(G,1)) = clg_t;

for part_i = 1:size(G,1)
    if G(part_i,1) == 0
        % only parent is class variable
        % y,x,alpha
        for class_i = 1:K
            % y
            [mu_y sigma_y] = FitGaussianParameters(dataset(labels(:,class_i)==1,part_i,1));
            P.clg(part_i).mu_y(class_i) = mu_y;
            P.clg(part_i).sigma_y(class_i) = sigma_y;
            % x
            [mu_x sigma_x] = FitGaussianParameters(dataset(labels(:,class_i)==1,part_i,2));
            P.clg(part_i).mu_x(class_i) = mu_x;
            P.clg(part_i).sigma_x(class_i) = sigma_x;
            % alpha
            [mu_angle sigma_angle] = FitGaussianParameters(dataset(labels(:,class_i)==1,part_i,3));
            P.clg(part_i).mu_angle(class_i) = mu_angle;
            P.clg(part_i).sigma_angle(class_i) = sigma_angle;
        end
    elseif G(part_i,1) == 1
        % parent: 1) class variable, 2) another part
        for class_i = 1:K
            
            U = dataset(labels(:,class_i)==1,1,:);
            U = reshape(U,size(U,1),size(U,3));
            % y
            [Beta sigma_y] = FitLinearGaussianParameters( dataset(labels(:,class_i)==1,part_i,1), U );
            P.clg(part_i).sigma_y(class_i) = sigma_y;
            % need to put Beta(0) in front
            Beta = [Beta(end) Beta(1:end-1)'];
            P.clg(part_i).theta(class_i,1:4) = Beta;
            % x
            [Beta sigma_x] = FitLinearGaussianParameters( dataset(labels(:,class_i)==1,part_i,2), U );
            P.clg(part_i).sigma_x(class_i) = sigma_x;
            % need to put Beta(0) in front
            Beta = [Beta(end) Beta(1:end-1)'];
            P.clg(part_i).theta(class_i,5:8) = Beta;
            % alpha
            [Beta sigma_angle] = FitLinearGaussianParameters( dataset(labels(:,class_i)==1,part_i,3), U );
            P.clg(part_i).sigma_angle(class_i) = sigma_angle;
            % need to put Beta(0) in front
            Beta = [Beta(end) Beta(1:end-1)'];
            P.clg(part_i).theta(class_i,9:12) = Beta;
        end
    else
        fprintf(1,'error');
    end
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);

fprintf('log likelihood: %f\n', loglikelihood);

