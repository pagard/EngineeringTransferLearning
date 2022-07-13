function Ytest = k_means_predict(classifier,Xtest,plt)
% predict k-means clustering
%
% Inputs
% Xtest = test inputs (n*d)
% classifier = classifier parameters from k-means
% plt = 0 - no plot
%
% Output
% Ytest = label predictions
%
% Paul Gardner, University of Sheffield 2022

if nargin<3
    plt = 0; % default no plot
end

% Label data according to distance from means
D = pdist2(Xtest,classifier.mu);
[~,ind] = min(D,[],2);
Ytest = ind; % label predictions

% plot if 2D and plot on
if plt ~= 0 && size(Xtest,2) == 2
    figure(100)
    hold off
    gscatter(Xtest(:,1),Xtest(:,2),Ytest)
    hold on
    plot_gaussian_2d(classifier.mu,...
        repmat(eye(size(Xtest,2)),1,1,classifier.k));
end