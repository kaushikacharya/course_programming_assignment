%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% flag for each factor to keep track whether it has been assigned to any
% clique or not. Note: a factor can be assigned to only one clique
% flag_factor_assignment_to_clique = zeros(1,length(C.factorList));

% https://class.coursera.org/pgm/forum/thread?thread_id=1352
% helpful comments:
% 1. Posted by Toshiaki Takeuchi (Student) on Thu 12 Apr 2012 5:11:17 PM PDT 
% 2. Posted by Leonardo Couto (Student) on Thu 12 Apr 2012 8:47:42 PM PDT 

% factors assigned to each clique
clique_factor_list = cell(1,length(C.nodes));

for fact_i = 1:length(C.factorList)
    % if ~flag_factor_assignment_to_clique(fact_i)
        % current factor not yet assigned to any clique
        for clique_j = 1:length(C.nodes)
            if all(ismember(C.factorList(fact_i).var,C.nodes{clique_j}))
                clique_factor_list{clique_j}(end+1) = fact_i;
                break; % Note: a factor can be assigned to only one clique
            end
        end
    % end
end

var_array = [C.factorList.var];
card_array = [C.factorList.card];
    
for i = 1:length(C.nodes)
    P.cliqueList(i).var = C.nodes{i};
    % P.cliqueList(i).card = C.card(C.nodes{i});
    
    for var = P.cliqueList(i).var
        [junk,index] = find(var_array == var,1);
        P.cliqueList(i).card(end+1) = card_array(index);
    end
    % first, create initial factors for each clique before assigning
    % factors to cliques
    factProd.var = C.nodes{i};
    factProd.card =  P.cliqueList(i).card; % C.card(C.nodes{i});
    factProd.val = ones(1,prod(factProd.card));
    % now take the factors one by one
    for fact_i = clique_factor_list{i}
        factProd = FactorProduct(factProd,C.factorList(fact_i));
    end
    P.cliqueList(i).val = factProd.val;
end

% now set the edges between cliques
P.edges = C.edges;
% for i = 1:length(C.nodes)
%     for j = i+1:length(C.nodes)
%         if any(ismember(C.nodes{i},C.nodes{j}))
%             P.edges(i,j) = 1;
%             P.edges(j,i) = 1;
%         end
%     end
% end

%% CHECK
% if length(P.cliqueList) ~= length(InitPotential.RESULT.cliqueList)
%     fprintf(1,'mismatch in length of cliqueList');
% end
% 
% for i = 1:length(P.cliqueList)
%     if ~all(P.cliqueList(i).var == InitPotential.RESULT.cliqueList(i).var)
%         fprintf(1,'mismatch in var field in i = %d\n',i);
%     end
%     if ~all(P.cliqueList(i).card == InitPotential.RESULT.cliqueList(i).card)
%         fprintf(1,'mismatch in card field in i = %d\n',i);
%     end
%     if ~all(P.cliqueList(i).val == InitPotential.RESULT.cliqueList(i).val)
%         fprintf(1,'mismatch in val field in i = %d\n',i);
%     end
% end

end

