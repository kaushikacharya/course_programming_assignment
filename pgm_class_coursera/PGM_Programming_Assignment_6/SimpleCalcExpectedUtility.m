% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);
  EU = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % TO DO: not yet handled when parents of random variable X also included
%  C = FactorProduct(I.RandomFactors, I.DecisionFactors);
%  EU = C.val*U.val'; 
  
    % setdiff(unique([F.var]),U.var) -> eliminate the parents of X
%   E = VariableElimination(F, setdiff(unique([F.var]),U.var)) ;
%   C = FactorProduct(E(1),E(2));
%   EU = C.val*U.val'; 

% https://class.coursera.org/pgm/forum/thread?thread_id=1790
% Posted by Kleyson de Sousa Rios (Student)
% on Thu 3 May 2012 10:57:48 AM PDT
% plus correction by Vladimir
factProd = F(1);
for i = 2:length(F)
    factProd = FactorProduct(factProd,F(i));
end
% E = VariableElimination(factProd, setdiff(factProd.var,U.var)) ;
% EU = E.val*U.val';

factProd = FactorProduct(factProd,U);
E = VariableElimination(factProd, setdiff(factProd.var,I.DecisionFactors.var));
EU = sum(E.val);

% https://class.coursera.org/pgm/forum/thread?thread_id=2085
% another approach : explanation by Posted by Quentin Pradet (Student)
% on Sun 6 May 2012 6:36:50 AM PDT 

end
