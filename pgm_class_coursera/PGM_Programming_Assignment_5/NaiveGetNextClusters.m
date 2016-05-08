%NAIVEGETNEXTCLUSTERS Takes in a node adjacency matrix and returns the indices
%   of the nodes between which the m+1th message should be passed.
%
%   Output [i j]
%     i = the origin of the m+1th message
%     j = the destination of the m+1th message
%
%   This method should iterate over the messages in increasing order where
%   messages are sorted in ascending ordered by their destination index and 
%   ties are broken based on the origin index. (note: this differs from PA4's
%   ordering)
%
%   Thus, if m is 0, [i j] will be the pair of clusters with the lowest j value
%   and (of those pairs over this j) lowest i value as this is the 'first'
%   element in our ordering. (this difference is because matlab is 1-indexed)
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function [i, j] = NaiveGetNextClusters(P, m)

    i = size(P.clusterList,1);
    j = size(P.clusterList,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    % Find the indices between which to pass a cluster
    % The 'find' function may be useful
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % https://class.coursera.org/pgm/forum/thread?thread_id=1455
    % A clique can't send a message at any time, but this function isn't
    % dealing with cliques. It is dealing with clusters. 
    % Posted by Anne Paulson (Student)
    array_cumsum = cumsum(sum(P.edges,1));
    n_msg_passed_in_full_cycle = sum(sum(P.edges,1));
    msg_in_cur_cycle_index = mod(m,n_msg_passed_in_full_cycle)+1;
    j = find(msg_in_cur_cycle_index <= array_cumsum,1);
    row_array = find(P.edges(:,j) == 1);
    
    if j == 1
        i = row_array(msg_in_cur_cycle_index);
    else
        i = row_array(msg_in_cur_cycle_index - array_cumsum(j-1));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

