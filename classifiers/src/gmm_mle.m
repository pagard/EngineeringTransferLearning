function [classifier] = gmm_mle(X,y,plt)
% Supervised Gaussian mixture model using maximum likelihood estimates
%
% Inputs
% X = inputs (n*d)
% y = labels (n*1)
% plt = 0 - no plot
%
% Outputs
% classifier = structure of GMM parameters
%   classifier.mu = means (K*d)
%   classifier.sigma = covariances (d*d*K)
%   classifier.lambda = mixing proportions (K)
%   classifier.K = no. of components 
%   classifier.classes = list of class labels (K)
% 
% Paul Gardner, University of Sheffield 2022

if nargin<3
    plt = 0; % default, no plot
end

[n,d] = size(X); % size of data

% responsibility matrix
labs = unique(y); % unique labels
K = length(labs); % no. of components
r = (y==labs'); % responsibility matrix

rk = sum(r,1); % class counts
lambda = rk./n; % mixing proportions

mu = r'*X./rk'; % means
% covariance
sigma = nan(d,d,K);
for k = 1:K    
    sigma(:,:,k) = 1./rk(k)*(r(:,k).*X)'*X ...
        - mu(k,:)'*mu(k,:); 
end

% plot if 2D and plot on
if plt ~= 0 && d == 2
    figure(100)
    hold off
    gscatter(X(:,1),X(:,2),y)
    hold on
    plot_gaussian_2d(mu,sigma);
end

% pack classifier
classifier.mu = mu;
classifier.sigma = sigma;
classifier.lambda = lambda;
classifier.K = K;
classifier.classes = labs;