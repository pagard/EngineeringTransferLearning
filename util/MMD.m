function [mmd,mmd_c] = MMD(Xs,Xt,kern,hyp,Ys,Yt)
% Maximum mean discrepancy distance
%
% Inputs
% Xs = source data (ns*d)
% Xt = target data (nt*d)
% kern = function handle of kernel function or no function handle for X
% hyp = kernel hyperparameters
% Ys = source labels (ns*1)
% Yt = target labels (nt*1)
%
% Output
% mmd = marginal maximum mean discrepancy
% mmd_c = approx. of of maximum mean discrepancy (only if Ys and Yt are
% given)
%
% Paul Gardner, University of Sheffield 2022

X = [Xs; Xt];
ns = size(Xs,1);
nt = size(Xt,1);
n = ns+nt;

% calculate M - constants from MMD biased v-statistic
M0 = zeros(n) -1/(ns*nt);
M0(1:ns,1:ns) = 1/ns^2;
M0(ns+1:end,ns+1:end) = 1/nt^2;

if isa(kern,'function_handle')
    % Calculate kernel matrix
    K = kern(hyp,X,X);
    K = K + eye(n)*1e-6;
else
    % No kernel embedding
    K = X;
    n = d;
end

% marginal mmd
mmd = sum(diag(K*M0));

M = M0; % initalise

if nargin == 6
    for c = unique(Ys)'
        
        cYs = find(c==Ys); % index's of Xs's in class c
        cYt = find(c==Yt); % index's of Xt's in class c
        
        % Mc
        nsc = length(cYs); % number of source inputs in the class
        ntc = length(cYt); % number of target inputs in the class
        
        % Create MMD matrix
        M(cYs,cYs) = M(cYs,cYs) + 1/nsc^2;
        M(ns+cYt,ns+cYt) = M(ns+cYt,ns+cYt) + 1/ntc^2;
        M(cYs,ns+cYt) = M(cYs,ns+cYt) -1/(nsc*ntc);
        M(ns+cYt,cYs) = M(ns+cYt,cYs) -1/(nsc*ntc);
    end
    
    % approx. of joint mmd
    mmd_c = sum(diag(K*M));
end