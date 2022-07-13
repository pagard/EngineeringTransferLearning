function [H,lambda_mle,mu_s,sigma_s,lambda_s,lml] = da_gmm_em(Xs,ys,Xt,H,tol,method,k)
% Expectation maximisation (EM) domain-adapted Gaussian mixture model
% 
% Inputs
% Xs = source features (ns*d)
% Ys = source labels (ns*1)
% Xt = target features (nt*d)
% H = mapping matrix (d*d)
% tol = convergence tolerance on change in likelihood in EM
% method = covariance type, 1 = different covariances 2 = same covariances
% k = no. of components in mixture model, if ys is not empty k does not
% need to be specified

[n,D] = size(Xt); % size of target data

if nargin<6
    method = 1; % default method is different covariances
end

if ~isempty(ys)
    % fit source GMM parameters
    k = length(unique(ys)); % number of components
    
    [prior.mu0,prior.S0] = gmm_mle(Xs,ys);
    [mu_s,sigma_s,lambda_s] = gmm_map(Xs,ys,prior);
else
    if nargin < 7
        k = input('K =');
    end
    [mu_s,sigma_s,lambda_s] = gmm_mle_em(Xs,k,1e-9);
end

if nargin<5
    H = eye(D,D); % initialise projection
end
if nargin<6
    tol = 1e-4;
end

% mixing proportion
lambda_mle = ones(1,k)/k;

log_lik = [];
r = nan(n,k);

while length(log_lik)<4 || abs(sum(log_lik(end)-log_lik(end-2:end))) > tol
    
    % E-step
    lml = 0;
    log_r = nan(n,k);
    for i = 1:n_u
        pi_Nk = nan(1,k);
        r_pi_Nk = nan(1,k);
        for j = 1:k
            pi_Nk(j) = log(lambda_mle(j)) + ...
                lnmvnpdf(Xt(ind_u(i),:)*H,mu_s(j,:),sigma_s{j}); % pi_k*N_k(x_i|mu_k,sigma_k)
            r_pi_Nk(j) = r(i,j)*pi_Nk(j);
        end
        log_r(ind_u(i),:) = pi_Nk - log(sum(exp(pi_Nk-max(pi_Nk)))) - max(pi_Nk); % log responsibility using log sum exp trick
        lml = lml + log(sum(exp(r_pi_Nk))); % log likelihood
    end
    r = exp(log_r); % responsibility r: p(y=k | x, D)
    r(isnan(r(:,1)),:) = r_lab(any(r_lab~=0,2),:);
    
    % M-step
    lambda_mle = mean(r); % update mixing proportions
    
    % equal covariance
    if method ~= 1
        H = (mu_s'*r'*Xt)/(Xt'*Xt + eye(D)*1e-4); % projection matrix if covariances are equal
        cost = costfn(reshape(H,D*D,1),Xt,Xs,mu_s,sigma_s,r);
        Xhat = Xt*H; % projected data
    else
        H0 = reshape(H,D*D,1); % pack up last guess as initial guess
        [h_opt,cost] = fminsearch(@(h) costfn(h,Xt,Xs,mu_s,sigma_s,r),H0); % optimise
        H = reshape(h_opt(1:D*D),D,D); % unpack optimised matrix
        Xhat = Xt*H; % projected data
    end
    
    % calculate the actual likelihood
    
    log_lik = [log_lik, -lml];
    
    % monitor convergence
%     if D == 2
%         figure(10)
%         subplot(1,3,1)
%         plot(1:length(log_lik),log_lik,'k-')
%         subplot(1,3,2:3)
%         gscatter(Xs(:,1),Xs(:,2),ys,[],'.',1)
%         hold on
%         gscatter(Xhat(:,1),Xhat(:,2),yt,[],'+')
%         plot_gaussian(mu_s,sigma_s,50);
%         hold off
%         pause(0.1)
%     else
%         figure(10)
%         plot(1:length(log_lik),log_lik,'k-')
%     end
    
end

% Max lml
lml = 0;
log_r = nan(n,k);
for i = 1:n_u
    pi_Nk = nan(1,k);
    r_pi_Nk = nan(1,k);
    for j = 1:k
        pi_Nk(j) = log(lambda_mle(j)) + ...
            lnmvnpdf(Xt(ind_u(i),:)*H,mu_s(j,:),sigma_s{j}); % pi_k*N_k(x_i|mu_k,sigma_k)
        r_pi_Nk(j) = r(i,j)*pi_Nk(j);
    end
    log_r(ind_u(i),:) = pi_Nk - log(sum(exp(pi_Nk-max(pi_Nk)))) - max(pi_Nk); % log responsibility using log sum exp trick
    lml = lml + log(sum(exp(r_pi_Nk))); % log likelihood
end

% if D == 2
%     figure
%     gscatter(Xs(:,1),Xs(:,2),ys,[],'+',5)
%     hold on
%     gscatter(Xhat(:,1),Xhat(:,2),yt,[],'.')
%     plot_gaussian(mu_s,sigma_s,50);
% end

end

%% E_q(H)

function cost = costfn(H0,X,Xs,mu,sigma,r)

[n,D] = size(X);
k = size(mu,1);

H = reshape(H0,D,D); % unpack vector into matrix

Xhat = X*H;

cost = nan(k,1);
for j = 1:k
    xdiff = bsxfun(@minus,Xhat,mu(j,:));
    R = cholcov(sigma{j});
    xRinv = xdiff/R;
    dist = sum(xRinv.^2, 2);
    cost(j) = sum(r(:,j).*dist);
end
cost = sum(cost);

cost = cost;% + abs(sum(sum(Xhat'*Xhat - Xs'*Xs)));% + trace(H'*H); % regularise

end