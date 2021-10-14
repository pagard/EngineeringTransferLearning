function [Zs,Zt,W,mmd] = tca(Xs,Xt,kern,hyp,mu,k)
% Transfer component anaylsis
%
% Inputs
% Xs = source features (ns*d)
% Xt = target features (nt*d)
% kern = function handle of kernel function or no function handle for X
% hyp = kernel hyperparameters
% mu = trade-off parameter
% k = no. of eigenvectors (dimension of embedding <= d)
%
% Outputs
% Zs = k-dimension transformed source feature space (ns*k)
% Zt = k-dimension transformed target feature space (nt*k)
% W = k-dimension embedding matrix (or adaptation matrix)
% mmd = mmd distance of transformed space
%
% Paul Gardner, Sheffield University 2019

ns = size(Xs,1); % length of source data
nt = size(Xt,1); % length of target data

X = [Xs;Xt]; % combine data
[n,d] = size(X); % length and dimension of total data set

if k>d; error('k must be less than d'); end

% calculate M - constants from MMD biased v-statistic
M = zeros(n) -1/(ns*nt);
M(1:ns,1:ns) = 1/ns^2;
M(ns+1:end,ns+1:end) = 1/nt^2;

% calculate H
H = eye(n) - ones(n)/(n); % centering matrix

% kernel embedding
if isa(kern,'function_handle')
    % Calculate kernel matrix
    K = kern(hyp,X,X);
else
    % No kernel embedding
    K = X;
    n = d;
end

% k smallest eigenvectors
[W,~] = eigs(mu*eye(n)+K'*M*K, K'*H*K,k,'sm');
Z = K*W;

% extract transformed source and target data
Zs = Z(1:ns,:);
Zt = Z(ns+1:ns+nt,:);

% mmd of transfer space
mmd = sum(diag(W'*K*M*K*W));

end