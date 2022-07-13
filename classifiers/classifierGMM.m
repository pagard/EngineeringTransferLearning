function [Ytp,classifier] = classifierGMM(Zs,Ys,Zt,classifier)
% (Supervised)Gaussian mixture model (maximum likelihood estimates)
%
% Inputs 
% Zs = source data
% ks = no. of source components
% Zt = target data
% classifier = pretrained classifier
%
% Outputs
% Ytp = target label predictions
% classifier = trained classifier

if nargin <4
    % Train supervised GMM Classifier
    classifier = gmm_mle(Zs,Ys);
end

% Predict
Ytp = gmm_mle_predict(classifier,Zt);

end
