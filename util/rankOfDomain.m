function ROD = rankOfDomain(Ps,Pt,Xs,Xt,d)
% Rank of domain
%
% Inputs
% Ps = soruce subspace basis
% Pt = target subspace basis
% Xs = source dataset
% Xt = target dataset
% d = dimension of PCA subspaces
%
% Output
% ROD = rank of domain
%
% Paul Gardner, University of Sheffield 2022

% principal angles from SVD of inner product between subspaces (S and
% T)
[U,S,V] = svd(Ps(:,1:d)'*Pt(:,1:d));
th = real(acos(diag(S))); % principal angles

% convert bases - principal vectors
s = Ps(:,1:d)*U;
t = Pt(:,1:d)*V;

% center data for zero-mean KL
Xss = Xs - mean(Xs);
Xtt = Xt - mean(Xt);

% PCA covariances
Sig2s = diag(1/size(Xss,1).*((s'*Xss')*Xss*s));
Sig2t = diag(1/size(Xtt,1).*((t'*Xtt')*Xtt*t));

ROD = (1/d)*sum(th.*(0.5.*Sig2s./Sig2t + 0.5.*Sig2t./Sig2s - 1)); % rank of domain

end