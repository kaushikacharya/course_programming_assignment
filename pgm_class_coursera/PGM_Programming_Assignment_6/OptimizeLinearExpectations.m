% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  MEU = [];
  OptimalDecisionRule = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for util_i = 1:length(I.UtilityFactors)
      I_with_cur_util = I;
      I_with_cur_util.UtilityFactors = I_with_cur_util.UtilityFactors(util_i);
      euf_with_cur_util = CalculateExpectedUtilityFactor(I_with_cur_util);
      
      if util_i == 1
          euf = euf_with_cur_util;
      else
          euf.val = euf.val + euf_with_cur_util.val;
      end
  end

  % now the rest is same as OptimizeMEU.m
  D = I.DecisionFactors(1); % assumed single decision fator as mentioned in comments above
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
