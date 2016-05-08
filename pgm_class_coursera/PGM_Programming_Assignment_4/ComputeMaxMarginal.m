function M = ComputeMaxMarginal(V, F, E)

% Check for empty factor list
assert(numel(F) ~= 0, 'Error: empty factor list');

F = ObserveEvidence(F, E);
Joint = ComputeJointDistribution(F);
% val are in log-space, hence normalization not needed
% mentioned in assigment description .pdf
% Joint.val = Joint.val ./ sum(Joint.val);
M = FactorMaxMarginalization(Joint, setdiff(Joint.var, V));
%M.val = M.val ./ sum(M.val);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end