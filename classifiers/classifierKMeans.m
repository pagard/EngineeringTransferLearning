function [Ytp,classifier] = classifierKMeans(Zs,ks,Zt,classifier)
% (Unsupervised) k-means classifier
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
    % Train k-means Classifier
    classifier = k_means(Zs,ks);
end

% Predict
Ytp = k_means_predict(classifier,Zt);

end
