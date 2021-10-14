function [Z] = domainAdaptationTransform(Xtest,Xs,Xt,W,kern,hyp)
% Transforms inputs based on learnt weigths and kernel in the form Z=KW
%
% Inputs
% Xtest = new feature data to transform (ntest*d)
% Xs = source features (ns*d)
% Xt = target features (nt*d)
% W = weights from kernel embedding (ns+nt*k)
% kern = function handle of kernel function
% hyp = kernel hyperparameters
%
% Output
% Z = transformed input vector
%
% Paul Gardner, Sheffield University 2019

X = [Xs;Xt]; % combine training data

if isa(kern,'function_handle')
    
    if strcmp(func2str(kern),'kernelRBF_median')
        [~,hyp] = kern(hyp,X,X);
        K = kernelRBF(hyp,Xtest,X);
    else
        % Calculate kernel matrix
        K = kern(hyp,Xtest,X);
    end
else
    error('kern must be a function handle for a kernel function');
end

Z = K*W; % transform k-components for test data

end
