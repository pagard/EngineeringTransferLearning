function [classifier,Y] = gmm_mle_em(X,K,tol,method,plt)
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
    
    % randomly assign initial guess to mean 
    mu = X(ind(1:K),:);
else
    % k-means initialise
    kmeans = k_means(X,K,1);
    mu = kmeans.mu;
end
sigma = repmat(eye(d),1,1,K); % initial unity covariance

% mixing proportion
lambda = ones(1,K)/K;

lml = [];
while length(lml)<3 || abs(sum(lml(end)-lml(end-2:end))) > tol

    % E-step  
    
    % Gaussian likelihood per class
    ln_lik = nan(n,K);
    for k = 1:K
        ln_lik(:,k) = lnmvnpdf(X,mu(k,:),...
            sigma(:,:,k)); % N_k(x_i|mu_k,sigma_k)
    end
    ln_lik = log(lambda) + ln_lik; % mixture likelihood, pi_k*N_k(x_i|mu_k,sigma_k)

    ln_r = ln_lik - log(sum(exp(ln_lik - max(ln_lik,[],2)),2)) ...
        - max(ln_lik,[],2); % log responsibilities
    r = exp(ln_r); % responsibilities, p(y | x, D)
    
    [~,Y] = max(r,[],2); % label predictions (mle) 
    
    lml = [lml, sum(log(sum(exp(ln_lik),2)))]; % log likelihood
    
    % M-step
    
    % update mixture components
    lambda = mean(r); 
    
    Nk = sum(r,1); % counts
    
    % update mean
    mu = r'*X./Nk';
    
    for k = 1:K
        sigma(:,:,k) = 1/Nk(k)*(r(:,k).*(X-mu(k,:)))'*(X-mu(k,:)); % covariance update
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
        plot(1:length(lml),lml,'k-') % plot likelihood for
        % convergence monitoring
        xlabel('Iteration')
        ylabel('Log Likelihood')
    end
    
end

% pack classifier
classifier.mu = mu;
classifier.sigma = sigma;
classifier.lambda = lambda;
classifier.K = K;