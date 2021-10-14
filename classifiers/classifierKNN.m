function [Ytp,classifier] = classifierKNN(Zs,Ys,Zt,classifier)
% KNN with number of neighbours equal to the number of classes - 1
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
    noNeigh = max(Ys)-1;
    % Train KNN
    classifier = fitcknn(Zs,Ys,'NumNeighbors',noNeigh);
end

% Predict
Ytp = predict(classifier,Zt);

end
