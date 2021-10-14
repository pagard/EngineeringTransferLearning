function [Ytp,classifier] = classifierKNN_cv(Zs,Ys,Zt,classifier)
% KNN with number of neighbour cross validated
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
    % Train KNN
    classifier = fitcknn(Zs,Ys,'optimizeHyperparameters',...
        'NumNeighbors',...
        'HyperparameterOptimizationOptions',...
        struct('Verbose',0,'ShowPlots',false,'UseParallel',true));
end

% Predict
Ytp = predict(classifier,Zt);

end
