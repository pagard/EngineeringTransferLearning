function K = kernelLinear(hyp,xp,xq)
% Linear kernel
%
% Inputs
% hyp = blank
% xp, xq = data in kernel
%
% Ouput
% K = kernel matrix

K = xp*xq';