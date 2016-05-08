%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code
M = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step #1: create clique tree from the list of factors
P = CreateCliqueTree(F, E);
% step #2: do the message passing for calibration
P = CliqueTreeCalibrate(P, isMax);

% now compute the final marginal for each variable
unique_var = unique([F.var]);
flag_final_margnal_array = zeros(1,length(unique_var)); % to keep track of which variable has been done

M = repmat(struct('var', [], 'card', [], 'val', []), 1, length(unique_var));

for clique_i = 1:length(P.cliqueList)
    for var = P.cliqueList(clique_i).var
        if ~flag_final_margnal_array(unique_var == var)
            if ~isMax
                marginal_var = ComputeMarginal(var, P.cliqueList(clique_i),E);
                marginal_var.val = marginal_var.val/sum(marginal_var.val); % this step not needed. ComputeMarginal is already doing normalization.
            else
                marginal_var = ComputeMaxMarginal(var, P.cliqueList(clique_i),E);
            end
            M(unique_var == var) = marginal_var;
            % set the flag
            flag_final_margnal_array(unique_var == var) = true;
        end
    end
end

%%CHECK
% if length(M) ~= length(ExactMarginal.RESULT)
%     fprintf(1,'mismatch in length of var');
% end
% 
% for i = 1:length(M)
%     if ~all(M(i).var == ExactMarginal.RESULT(i).var)
%         fprintf(1,'mismatch in var field in i = %d\n',i);
%     end
%     if ~all(M(i).card == ExactMarginal.RESULT(i).card)
%         fprintf(1,'mismatch in card field in i = %d\n',i);
%     end
%     if ~all(M(i).val == ExactMarginal.RESULT(i).val)
%         fprintf(1,'mismatch in val field in i = %d\n',i);
%     end
% end

end
