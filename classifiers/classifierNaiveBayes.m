function [Ytp,classifier,posteriors] = classifierNaiveBayes(Zs,Ys,Zt,classifier)
% Naive Bayes classifier
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
    % Train Naive Bayes Classifier
    classifier = fitcnb(Zs,Ys);
end

% Predict
[Ytp,posteriors] = predict(classifier,Zt);

end
