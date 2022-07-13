function [Ytp,classifier,posteriors] = classifierGMM_EM(Zs,Ys,Zt,classifier)
% (Unsupervised) Expectation maximisation (EM) Gaussian mixture model
%
% Inputs 
% Zs = source data
% Ys = source labels
% Zt = target data
% classifier = pretrained classifier
%
% Outputs
% Ytp = target label predictions
% classifier = trained classifier

if nargin <4
    % Train GMM Classifier
    
end

% Predict
[Ytp,posteriors] = predict(classifier,Zt);

end
