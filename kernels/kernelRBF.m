function [K,hyp] = kernelRBF(hyp,xp,xq)
% Radial basis function kernel
%
% Inputs
% hyp = sigma, if hyp=nan median hueristic is used
% xp, xq = data in kernel
%
% Ouput
% K = kernel matrix

if isnan(hyp)
    % Median hueristic for hyperparameter
    Z = [xp;xq];
    dist = pdist2(Z,Z).^2; % distances between data
    dist = reshape(triu(dist),size(Z,1).^2,1); % upper triangular (so no repeats)
    hyp = sqrt(0.5*median(dist(dist>0)));
end

d = pdist2(bsxfun(@rdivide,xp,sqrt(2*hyp^2)),bsxfun(@rdivide,xq,sqrt(2*hyp^2))).^2;
K = exp(-d);
