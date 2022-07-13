function [classifier,Y] = gmm_mle_em(X,k,tol,method,plt)
% Unsupervised Gaussian mixture model using EM and maximum likelihood
% estimates
%
% Inputs
% X = inputs (n*d)
% k = no. of components
% tol = convergence tolerance on means
% method = initialisation method, 0 - random, 1 - k-means
% plt = 0 - no plot 
%
% Outputs
% classifier = structure of GMM parameters
%   classifier.mu = means (k*d)
%   classifier.sigma = covariances (d*d*k)
%   classifier.lambda = mixing proportions (k)
%   classifier.lml = complete log likelihood
% Y = label predictions
%
% Paul Gardner, University of Sheffield 2022

if nargin<3 || isempty(tol)
    tol = 1e-6; % default tolerance
end

if nargin<4
    method = 0; % default, random initialise
end

if nargin<5
    plt = 0; % default, no plot
end

[n,d] = size(X); % size of data

% initialise GMM
if method == 0
    % random initialise
    ind = randperm(n);
    
    % evenly split data randomly and assign initial guess and mean and
    % covariance
    mu = X(ind(1:k),:);
else
    % k-means initialise
    kmeans = k_means(X,k);
    mu = kmeans.mu;
end
sigma = repmat(eye(d),1,1,k);

% mixing proportion
lambda = ones(1,k)/k;

log_lik = [];
while length(log_lik)<3 || abs(sum(log_lik(end)-log_lik(end-2:end))) > tol
    
    % E-step
    lml = 0;
    log_r = nan(n,k);
    for i = 1:n
        pi_Nk = nan(1,k);
        for j = 1:k
            pi_Nk(j) = log(lambda_mle(j)) + ...
                lnmvnpdf(X(i,:),mu(j,:),sigma(:,:,j)); % pi_k*N_k(x_i|mu_k,sigma_k)
        end
        lml = lml + log(sum(exp(pi_Nk))); % log likelihood
        log_r(i,:) = pi_Nk - log(sum(exp(pi_Nk-max(pi_Nk)))) - max(pi_Nk); % log responsibility using log sum exp trick
    end
    r = exp(log_r); % responsibility r: p(y | x, D)
    
    log_lik = [log_lik, -lml]; % add new log likelihood

    % M-step
    lambda_mle = mean(r); % mixture component update
    
    for i = 1:k
        mu(i,:) = sum(r(:,i).*X)./sum(r(:,i)); % mean
        
        sigma_k = zeros(d,d);
        Xdiff = bsxfun(@minus,X,mu(i,:)); % difference between data and mean
        for j = 1:n
            sigma_k = sigma_k + r(j,i).*(Xdiff(j,:)'*Xdiff(j,:));
        end        
        sigma(:,:,i) = sigma_k./sum(r(:,i)); % covariance
    end
    
    % plot if plot on
    if plt ~= 0
        figure(100)
        if d == 2 % plot space if 2D
            subplot(1,2,2)
            hold off
            gscatter(X(:,1),X(:,2),Y)
            hold on
            plot_gaussian_2d(mu,sigma);
            subplot(1,2,1)
        end
        plot(1:length(log_lik),log_lik,'k-') % plot likelihood for 
        % convergence monitoring
    end
    
end