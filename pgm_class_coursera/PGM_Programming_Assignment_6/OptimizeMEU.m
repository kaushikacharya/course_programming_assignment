% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
  euf = CalculateExpectedUtilityFactor(I);
  [junk,max_index] = max(euf.val);
  
  OptimalDecisionRule.var = euf.var;
  OptimalDecisionRule.card = euf.card;
  OptimalDecisionRule.val = zeros(1,length(euf.val));
  
  IdxtoAsgn = IndexToAssignment(1:prod(OptimalDecisionRule.card),OptimalDecisionRule.card);
  
  if length(D.var) > 1
      % D.var(1) -> decision variable, rest parents
      % Note: in euf the variables are in sorted form
      decision_variable_val = IdxtoAsgn(max_index,D.var(1)==euf.var);
      OptimalDecisionRule.val = (IdxtoAsgn(:,D.var(1)==euf.var) == decision_variable_val)';
  else
      decision_variable_val = IdxtoAsgn(max_index);
      OptimalDecisionRule.val = (IdxtoAsgn == decision_variable_val)';
      OptimalDecisionRule.var = [OptimalDecisionRule.var 0]; % to match the output of Test Case 1
  end
  
  MEU = euf.val*OptimalDecisionRule.val';

end
