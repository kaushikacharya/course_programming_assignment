function factors = ChooseTopSimilarityFactors (allFactors, F)
% This function chooses the similarity factors with the highest similarity
% out of all the possibilities.
%
% Input:
%   allFactors: An array of all the similarity factors.
%   F: The number of factors to select.
%
% Output:
%   factors: The F factors out of allFactors for which the similarity score
%     is highest.
%
% Hint: Recall that the similarity score for two images will be in every
%   factor table entry (for those two images' factor) where they are
%   assigned the same character value.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

% If there are fewer than F factors total, just return all of them.
if (length(allFactors) <= F)
    factors = allFactors;
    return;
end

% Your code here:
% factors = allFactors; %%% REMOVE THIS LINE
sim_factor_array = zeros(1,length(allFactors));
for i = 1:length(allFactors)
    for assgn_i = 1:allFactors(i).card(1)
        assign_to_index = AssignmentToIndex([assgn_i assgn_i], allFactors(i).card);
        sim_factor_array(i) =  sim_factor_array(i) + allFactors(i).val(assign_to_index);
    end
end

[junk,sort_indices] = sort(sim_factor_array,'descend');
factors = allFactors(sort_indices(1:F));

end

