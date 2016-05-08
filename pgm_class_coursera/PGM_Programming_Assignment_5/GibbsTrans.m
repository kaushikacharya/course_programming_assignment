% GIBBSTRANS
%
%  MCMC transition function that performs Gibbs sampling.
%  A - The current joint assignment.  This should be
%      updated to be the next assignment
%  G - The network
%  F - List of all factors
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function A = GibbsTrans(A, G, F)

for i = 1:length(G.names)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    % For each variable in the network sample a new value for it given everything
    % else consistent with A.  Then update A with this new value for the
    % variable.  NOTE: Your code should call BlockLogDistribution().
    % IMPORTANT: you should call the function randsample() exactly once
    % here, and it should be the only random function you call.
    %
    % Also, note that randsample() requires arguments in raw probability space
    % be sure that the arguments you pass to it meet that criteria
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % https://class.coursera.org/pgm/forum/thread?thread_id=1504
    
    % Posted by Tomas Krupka (Student)
    % on Sat 14 Apr 2012 3:18:03 PM PDT 
    % The implementation uses a linear congruential rng, which ensures that
    % all of us get the same sequence of random numbers.
    
    % Posted by Allan Joshua (Community TA)
    % on Sat 14 Apr 2012 8:02:06 PM PDT
    % has supported Tomas's explanation.
    
    % before start of testing the sequence of A one should do randi('seed',1);
    % This is done in submit.m
    % ----------
    
    % https://class.coursera.org/pgm/forum/thread?thread_id=1740
    % Posted by Anonymous
    % on Sun 22 Apr 2012 2:40:46 PM PDT 
    % explains how to write the code.
    
    LogBS = BlockLogDistribution(i, G, F, A);
    A(i) = randsample(G.card(i), 1, true, exp(LogBS));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
