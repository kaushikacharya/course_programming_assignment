function [Beta sigma] = FitLinearGaussianParameters(X, U)

% Estimate parameters of the linear Gaussian model:
% X|U ~ N(Beta(1)*U(1) + ... + Beta(n)*U(n) + Beta(n+1), sigma^2);

% Note that Matlab/Octave index from 1, we can't write Beta(0).
% So Beta(n+1) is essentially Beta(0) in the text book.

% X: (M x 1), the child variable, M examples
% U: (M x N), N parent variables, M examples
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

M = size(U,1);
N = size(U,2);

Beta = zeros(N+1,1);
sigma = 1;

% collect expectations and solve the linear system
% A = [ E[U(1)],      E[U(2)],      ... , E[U(n)],      1     ; 
%       E[U(1)*U(1)], E[U(2)*U(1)], ... , E[U(n)*U(1)], E[U(1)];
%       ...         , ...         , ... , ...         , ...   ;
%       E[U(1)*U(n)], E[U(2)*U(n)], ... , E[U(n)*U(n)], E[U(n)] ]

% construct A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = NaN(N+1,N+1);
% row #1
for i = 1:N
    A(1,i) = FitGaussianParameters(U(:,i));
end
A(1,N+1) = 1;

% 2nd row onwards
for row_i = 2:N+1
    for col_i = 1:N
        A(row_i,col_i) = FitGaussianParameters(U(:,col_i).*U(:,row_i-1));
    end
    A(row_i,N+1) = FitGaussianParameters(U(:,row_i-1));
end

% B = [ E[X]; E[X*U(1)]; ... ; E[X*U(n)] ]

% construct B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = NaN(1,N+1);
B(1) = FitGaussianParameters(X);
for i = 1:N
    B(i+1) = FitGaussianParameters(X.*U(:,i));
end

% solve A*Beta = B
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Beta = A\B';

% then compute sigma according to eq. (11) in PA description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
var2 = 0;
for i = 1:N
    for j = 1:N
        var2 = var2 + Beta(i)*Beta(j)*( FitGaussianParameters(U(:,i).*U(:,j)) - ...
            FitGaussianParameters(U(:,i))*FitGaussianParameters(U(:,j)) );
    end
end
var1 = FitGaussianParameters(X.*X) - FitGaussianParameters(X)*FitGaussianParameters(X);

sigma = sqrt(var1 - var2);