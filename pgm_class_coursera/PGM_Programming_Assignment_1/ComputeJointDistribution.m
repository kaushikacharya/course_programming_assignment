%ComputeJointDistribution Computes the joint distribution defined by a set
% of given factors
%
%   Joint = ComputeJointDistribution(F) computes the joint distribution
%   defined by a set of given factors
%
%   Joint is a factor that encapsulates the joint distribution given by F
%   F is a vector of factors (struct array) containing the factors 
%     defining the distribution
%

function Joint = ComputeJointDistribution(F)

  % Check for empty factor list
  if (numel(F) == 0)
      warning('Error: empty factor list');
      Joint = struct('var', [], 'card', [], 'val', []);      
      return;
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE:
% Compute the joint distribution defined by F
% You may assume that you are given legal CPDs so no input checking is required.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Joint = struct('var', [], 'card', [], 'val', []); % Returns empty factor. Change this.

if (numel(F) == 1)
    Joint = F(1);
else
    Joint = FactorProduct(F(1),F(2));
    for i = 3:numel(F)
        Joint = FactorProduct(Joint,F(i));
    end
end

% var_card_array = [];
% for i = 1:length(F)
%     var_card_array = [var_card_array [F(i).var; F(i).card]];
% end
% [junk, index] = unique(var_card_array(1,:));
% 
% Joint.var = var_card_array(1,index);
% Joint.card = var_card_array(2,index);
% 
% assignments = IndexToAssignment(1:prod(Joint.card), Joint.card);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

