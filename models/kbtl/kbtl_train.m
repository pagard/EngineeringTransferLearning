function [params,Y] = kbtl_train(K,y,params)
% function trains kernelised bayesian transfer learning - multi-class
% classification

% rand('state', params.seed); %#ok<RAND>
%randn('state', params.seed); %#ok<RAND>
rng(3)

% sizes of domains
T = length(K); % no. of tasks
N = cellfun(@length,K,'Uni',0); % no. of points (K in R^NxN)

Ls = cellfun(@(c) size(c,2),y);
if all(Ls==1)
    
    % convert labels from numerics to N x L matrix of +1 and -1
    labs = unique(cat(1,y{:})); % get vector of unique labels
    L = length(labs); % number of labels
    Y = cellfun(@(c) ones(c,L),N,'Uni',0); % initalise label matrix
    
    % convert to (NxL) matrix of [-1 +1]'s - could be sped up?
    for t = 1:T
        for l = 1:L
            Y{t}(y{t}~=labs(l),l) = -1;
        end
    end
    
else
    Y = y; % labels in N x L matrix form
    L = size(Y{1},2); % number of labels
end

% unpack parameters
R = params.R; % dimension size of H (i.e. dimensionality reduction)
Hsig2 = params.Hsigma2; % prior variance of H

% initalise dimensionality reduction part

% approximate posterior of gamma shape parameters (alpha) and inialise gamma scale
% parameters (beta)
Lambda.alpha = cellfun(@(c) (params.lambda.alpha + 0.5)*ones(c,R),N,'Uni',0); % A hyperparameters
Lambda.beta = cellfun(@(c) params.lambda.beta*ones(c),N,'Uni',0); % A hyperparameters

% intialise A and H
A.mu = cellfun(@(c) randn(c,R),N,'Uni',0); % initalise A mu as random matrix
A.sig_diag = cellfun(@(c) ones(c,R),N,'Uni',0); % initalise diag of A sigma as unit variance
A.sig = cellfun(@(c) repmat(eye(c),[1 1 R]),N,'Uni',0); % initalise A sigma as unit variance
H.mu = cellfun(@(c) randn(R,c),N,'Uni',0); % initalise H mu as random matrix
H.sig = cellfun(@(c) eye(R),cell(1,T),'Uni',0); % initalise H sigma as unit variance

% initalise classifier part

% approximate posterior of gamma shape parameters (alpha) and inialise gamma scale
% parameters (beta)
gam.alpha = (params.gamma.alpha + 0.5)*ones(L,1); % b hyperparameters
eta.alpha = (params.eta.alpha + 0.5)*ones(R,L); % w hyperparameters
gam.beta = params.gamma.beta*ones(L,1); % b hyperparameters
eta.beta = params.eta.beta*ones(R,L); % w hyperparameters

% weights and bias
bw.mu = [zeros(1,L); randn(R,L)]; % initalise mean weights and bias
bw.sig = repmat(eye(R+1),[1,1,L]); % initalise weigths and bias variance

% intialise discriminative function
f.mu = cellfun(@(c1,c2) (abs(randn(c1,L)) + params.margin).*sign(c2),N,y,'Uni',0); % initalise function (make sure either side of margin)
f.sig = cellfun(@(c) ones(c,L),N,'Uni',0); % initalise unit variance

% get cross product K*K'
KKt = cellfun(@(c) c*c',K,'Uni',0);

% margins
labs_neg = cellfun(@(c) c<0,Y,'Uni',0); % find -1 labels
labs_pos = cellfun(@(c) c>0,Y,'Uni',0); % find +1 labels
lower_margin = cellfun(@(c1,c2) -1e40*c1 + +params.margin*c2,labs_neg,labs_pos,'Uni',0); % lower margin
upper_margin = cellfun(@(c1,c2) 1e40*c2 + -params.margin*c1,labs_neg,labs_pos,'Uni',0); % upper margin

% variation inference steps
for i = 1:params.iter
    
    % update dimensionality reduction part
    
    % update lamda beta - hyperparameters of projection prior
    Lambda.beta = cellfun(@(c1,c2) 1./(1./params.lambda.beta + 0.5*(c1.^2 + c2)),A.mu,A.sig_diag,'Uni',0);
    
    % update A - projection matrix
    for t = 1:T
        for s = 1:R
            % update A covariance
            A.sig{t}(:,:,s) = (diag(Lambda.alpha{t}(:,s).*Lambda.beta{t}(:,s)) + KKt{t}/Hsig2)\eye(N{t});
            A.sig_diag{t}(:,s) = diag(A.sig{t}(:,:,s)); % store diag
            % update A mean
            A.mu{t}(:,s) = A.sig{t}(:,:,s)*(K{t}*H.mu{t}(s,:)'/Hsig2);
        end
    end
    
    % update H - latent subspace
    H.sig = cellfun(@(c) (eye(R)/Hsig2 + bw.mu(2:R+1)*bw.mu(2:R+1)' + sum(bw.sig(2:R+1,2:R+1,:),3))\eye(R),cell(1,T),'Uni',0);
    H.mu = cellfun(@(c1,c2) c1'*c2/Hsig2,A.mu,K,'Uni',0);
    for ll = 1:L
        H.mu = cellfun(@(c1,c2,c3) c1 + bw.mu(2:R+1,ll)*c2(:,ll)' - repmat(bw.mu(2:R+1,ll)*bw.mu(1,ll) + bw.sig(2:R+1,1,ll),1,c3),H.mu,f.mu,N,'Uni',0);
    end
    H.mu = cellfun(@(c1,c2) c1*c2,H.sig,H.mu,'Uni',0);
    
    % update classifier part
    
    for ll = 1:L
        % update gamma beta - hyperparameter of bias prior
        gam.beta(ll) = 1/(1/params.gamma.beta + 0.5*(bw.mu(1,ll)^2 + bw.sig(1,1,ll)));
        % update eta beta - hyperparameters of weight prior
        eta.beta(:,ll) = 1./(1/params.eta.beta + 0.5*(bw.mu(2:R+1,ll).^2 + diag(bw.sig(2:R+1,2:R+1,ll))));
        
        % update bias and weights
        % bias and weight covariance
        bw.sig(:,:,ll) = [gam.alpha(ll)*gam.beta(ll),zeros(1,R);zeros(R,1),diag(eta.alpha(:,ll).*eta.beta(:,ll))]; % non-task dependant
        bwsig = cellfun(@(c1,c2,c3) [c1, sum(c2,2)';... - Could be removed from loop?
            sum(c2,2), c2*c2' + c1*c3],N,H.mu,H.sig,'Uni',0); % create addition for each task
        bw.sig(:,:,ll) = (bw.sig(:,:,ll) + sum(cat(3,bwsig{:}),3))\eye(R+1,R+1); % sum all tasks and find inverse
        % bias and weight mean
        bwmu = cellfun(@(c1,c2,c3) [ones(1,c1);c2]*c3(:,ll),N,H.mu,f.mu,'Uni',0); % create addition for each task
        bw.mu(:,ll) = bw.sig(:,:,ll)*sum(cat(2,bwmu{:}),2); % mean bias and weight
    end
    
    % update f
    q_f_mu = cellfun(@(c1,c2,c3) [ones(1,c1); c2]'*bw.mu,N,H.mu,'Uni',0); % posterior update of f mu
    % truncated normal
    alpha_tn = cellfun(@(c1,c2) c1 - c2,lower_margin,q_f_mu,'Uni',0); % alpha for truncated normal
    beta_tn = cellfun(@(c1,c2) c1 - c2,upper_margin,q_f_mu,'Uni',0); % beta for truncated normal
    Z = cellfun(@(c1,c2) normcdf(c1) - normcdf(c2),beta_tn,alpha_tn,'Uni',0); % normalising constant
    Z = cellfun(@(c) c + (c==0) ,Z,'Uni',0); % normalising constant
    % f mean and variance
    f.mu = cellfun(@(c1,c2,c3,c4) c1 + (normpdf(c2)-normpdf(c3))./c4,q_f_mu,alpha_tn,beta_tn,Z,'Uni',0); % f mean through truncated normal
    f.sig = cellfun(@(c1,c2,c3) 1 + (c1.*normpdf(c1) - c2.*normpdf(c2))./c3 - (normpdf(c1)-normpdf(c2)./c3).^2,alpha_tn,beta_tn,Z,'Uni',0); % f sigma through truncated normal
    
    % variational lower bound - check equations
%     %     logQLambda =
%     logQA = 0;
%     for t = 1:T
%         for s = 1:R
%             RAsig = chol(A.sig{t}(:,:,s));
%             logQA = logQA + 0.5*(R+1)*(1+log(2*pi)) + sum(log(diag(RAsig)));
%         end
%     end
%     RHsig = cellfun(@(c) chol(c),H.sig,'Uni',0);
%     logQH = cellfun(@(c) 0.5*(R+1)*(1+log(2*pi)) + sum(log(diag(c))),RHsig,'Uni',0);
%     logQH = sum(cat(1,logQH{:}),1);
%     
%     logQgamma = (gam.alpha-1)*psi(gam.alpha) - log(gam.beta) + gam.alpha + log(gamma(gam.alpha));
%     logQeta = sum((eta.alpha-1).*psi(eta.alpha) - log(eta.beta) + eta.alpha + log(gamma(eta.alpha)));
%     RbwSig = chol(bw.sig);
%     logQbw = 0.5*(R+1)*(1+log(2*pi)) + sum(log(diag(RbwSig)));
%     %     logQf =
%     %
%     logQtheta = logQA + logQH + logQgamma + logQeta + logQbw;
%     if mod(i,5) == 0
%         fprintf(1,'Iteration: %5d Lower Bound: %4.5f \n', i,logQtheta);
%     end
    if mod(i,10) == 0
        fprintf(1,'Iteration: %5d \n', i);
    end
        
end

% pack parameters
params.Lambda = Lambda;
params.A = A;
params.gamma = gam;
params.eta = eta;
params.bw = bw;

end