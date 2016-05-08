% CHECKCONVERGENCE Ascertain whether the messages indicate that we have converged
%   converged = CHECKCONVERGENCE(MNEW,MOLD) compares lists of messages MNEW
%   and MOLD.  If the values listed in any message differs by more than the 
%   value 'thresh' then we determine that convergence has not occured and 
%   return converged=0, otherwise we have converged and converged=1
%
%   The 'message' data structure is an array of structs with the following 3 fields:
%     -.var:  the variables covered in this message
%     -.card: the cardinalities of those variables
%     -.val:  the value of the message w.r.t. the message's variables
%
%   MNEW and MOLD are the message where M(i,j).val gives the values associated
%   with the message from cluster i to cluster j.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function converged = CheckConvergence(mNew, mOld);
converged = true;
thresh = 1.0e-6;
%converged should be 1 if converged, 0 otherwise.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% https://class.coursera.org/pgm/forum/thread?thread_id=2107
% Posted by Anonymous
% on Fri 4 May 2012 1:57:13 AM PDT 
for msg_i = 1:length(mNew)
    % message between two clusters
    if any(abs(mNew(msg_i).val - mOld(msg_i).val) > thresh)
        converged = false;
        break;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

return;
