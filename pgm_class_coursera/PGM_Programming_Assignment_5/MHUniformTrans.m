% MHUNIFORMTRANS
%
%  MCMC Metropolis-Hastings transition function that
%  utilizes the uniform proposal distribution.
%  A - The current joint assignment.  This should be
%      updated to be the next assignment
%  G - The network
%  F - List of all factors
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function A = MHUniformTrans(A, G, F)

% Draw proposed new state from uniform distribution
A_prop = ceil(rand(1, length(A)) .* G.card);

p_acceptance = 0.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% Compute acceptance probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% https://class.coursera.org/pgm/forum/thread?thread_id=2076

% use LogProbOfJointAssignment
% Posted by Alicja Droszcz (Student)
% on Mon 30 Apr 2012 2:31:41 AM PDT 

% you get the pi(x) and pi(x') from LogProbOfJointAssignment

% Explained by Alicja Droszcz
% current state = x = variable assignment A
% next state = x' = variable assignment A_prop :-)

pi_A = exp(LogProbOfJointAssignment(F, A));
pi_A_prop = exp(LogProbOfJointAssignment(F, A_prop));

p_acceptance = min(1,pi_A_prop/pi_A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Accept or reject proposal
if rand() < p_acceptance
    % disp('Accepted');
    A = A_prop;
end