function [classifier,Y] = k_means(X,K,tol,plt)
% k-means clustering
%
% Inputs
% X = inputs (n*d)
% K = no. of components
% tol = convergence tolerance on means
% plt = 0 - no plot
%
% Outputs
% classifier = structure of k-means parameters
%   classifier.mu = means of clusters (K*d)
% Y = label predictions
% 
% Paul Gardner, University of Sheffield 2022

if nargin<3
    tol = 0.1; % default tolerance
end

if nargin<4
    plt = 0; % default, no plot
end

[n,d] = size(X); % size of data

% random initialise
ind = randperm(n);
mu = X(ind(1:K),:); % randomly initialise means

mu_old = mu*10; % initalise previous mu
while sum(sum((mu_old - mu).^2)) > tol
    mu_old = mu;

    % Label data according to distance from means
    D = pdist2(X,mu);
    [~,ind] = min(D,[],2);
    Y = ind;
    
    % Calculate means based on labelled data
    for k = 1:K
        mu(k,:) = mean(X(Y==k,:));
    end
    
    % plot if 2D and plot on
    if plt ~= 0 && d == 2
        figure(100)
        hold off
        gscatter(X(:,1),X(:,2),Y)
        hold on
        plot_gaussian_2d(mu,repmat(eye(d),1,1,K));
    end
end

% pack up classifier parameters
classifier.mu = mu;
classifier.k = K;