%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if isMax
     for clique_i = 1:length(P.cliqueList)
         P.cliqueList(clique_i).val = log(P.cliqueList(clique_i).val);
     end
 end
 
% https://class.coursera.org/pgm/forum/thread?thread_id=1587
while true
    [clique_i, clique_j] = GetNextCliques(P, MESSAGES);
    
    if clique_i == 0 && clique_j == 0
        break;
    else
        [intersect_var,index_i,index_j] = intersect(P.cliqueList(clique_i).var,P.cliqueList(clique_j).var);
        % MESSAGES(clique_i,clique_j).var = intersect_var;
        % MESSAGES(clique_i,clique_j).card = P.cliqueList(clique_i).card(index_i);
        neighbour_source_array = find(P.edges(:,clique_i))';
        neighbours_to_be_considered = setdiff(neighbour_source_array,clique_j);
        if isempty(neighbours_to_be_considered)
            % clique_i is at leaf
            if ~isMax
                MESSAGES(clique_i,clique_j) = ComputeMarginal(intersect_var, P.cliqueList(clique_i),[]);
            else
                MESSAGES(clique_i,clique_j) = ComputeMaxMarginal(intersect_var, P.cliqueList(clique_i),[]);
            end
        else
            clique_k = neighbours_to_be_considered(1);
            if ~isMax
                factProd = MESSAGES(clique_k,clique_i);
                for clique_k = neighbours_to_be_considered(2:end)
                    factProd = FactorProduct(factProd,MESSAGES(clique_k,clique_i));
                end
                % now multiply with the clique potential
                factProd = FactorProduct(P.cliqueList(clique_i), factProd);
                MESSAGES(clique_i,clique_j) = ComputeMarginal(intersect_var, factProd, []);
            else
                factSum = MESSAGES(clique_k,clique_i);
                for clique_k = neighbours_to_be_considered(2:end)
                    factSum = FactorSum(factSum,MESSAGES(clique_k,clique_i));
                end
                % now add with the clique potential
                factSum = FactorSum(P.cliqueList(clique_i), factSum);
                MESSAGES(clique_i,clique_j) = ComputeMaxMarginal(intersect_var, factSum, []);
            end
        end
        
        if ~isMax
            % normalize message val
            MESSAGES(clique_i,clique_j).val = MESSAGES(clique_i,clique_j).val/sum(MESSAGES(clique_i,clique_j).val);
        end
        % don't need to normalize for max-sum as mentioned in assignment
        % description.
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for clique_i = 1:N
    neighbours = find(P.edges(:,clique_i))';
    
    clique_k = neighbours(1);
    
    if ~isMax
        factProd = MESSAGES(clique_k,clique_i);
        for clique_k = neighbours(2:end)
            factProd = FactorProduct(factProd,MESSAGES(clique_k,clique_i));
        end
        % now multiply with the clique potential
        P.cliqueList(clique_i) = FactorProduct(P.cliqueList(clique_i), factProd);
    else
        factSum = MESSAGES(clique_k,clique_i);
        for clique_k = neighbours(2:end)
            factSum = FactorSum(factSum,MESSAGES(clique_k,clique_i));
        end
        % now sum with the clique potential
        P.cliqueList(clique_i) = FactorSum(P.cliqueList(clique_i), factSum);
    end
end

return

end
