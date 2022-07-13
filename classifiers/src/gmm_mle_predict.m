function [Ytest,r] = gmm_mle_predict(classifier,Xtest,plt)
% Supervised Gaussian mixture model using maximum likelihood estimates
%
% Inputs
% Xtest = test inputs (n*d)
% classifier = classifier parameters from a maximum likelihood GMM
% plt = 0 - no plot
%
% Output
% Ytest = label predictions
% r = responsibility matrix (probability of each class)
%
% Paul Gardner, University of Sheffield 2022

if nargin<3
    plt = 0; % default, no plot
end

[n,d] = size(Xtest); % size of data

% Gaussian likelihood per class
ln_lik = nan(n,classifier.K);
for k = 1:classifier.K
    ln_lik(:,k) = lnmvnpdf(Xtest,classifier.mu(k,:),...
        classifier.sigma(:,:,k)); % N_k(x_i|mu_k,sigma_k)
end
ln_lik = log(classifier.lambda) + ln_lik; % mixture likelihood, pi_k*N_k(x_i|mu_k,sigma_k)

ln_r = ln_lik - log(sum(exp(ln_lik - max(ln_lik,[],2)),2)) ...
    - max(ln_lik,[],2); % log responsibilities
r = exp(ln_r); % responsibilities, p(y | x, D)

[~,Ytest] = max(r,[],2); % label predictions (mle)

if isfield(classifier,'classes')
    Ytest = classifier.classes(Ytest); % number as exisiting classes if known
end

% plot if 2D and plot on
if plt ~= 0 && d == 2
    figure(100)
    hold off
    gscatter(Xtest(:,1),Xtest(:,2),Ytest)
    hold on
    plot_gaussian_2d(classifier.mu,classifier.sigma);
end
