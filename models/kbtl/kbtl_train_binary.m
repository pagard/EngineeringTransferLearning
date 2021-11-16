function params = kbtl_train_binary(K,y,params)
% Binary kernelised bayesian transfer learning - training
%
% Inputs
% K = cell of kernels from each domain
% y = cell of binary labels ([-1 or 1]) for each domain
% params = a structure of hyperparameters
%          params.lambda.kappa = shape hyperparameter of gamma prior for
%          projection matrices
%          params.lambda.theta = shape hyperparameter of gamma prior for
%          projection matrices
%          params.gamma.kappa = shape hyperparameter of gamma prior for
%          bias
%          params.gamma.theta = shape hyperparameter of gamma prior for
%          bias
%          params.eta.kappa = shape hyperparameter of gamma prior for
%          weights
%          params.eta.theta = shape hyperparameter of gamma prior for
%          weights
%          params.iter = number of iterations
%          params.margin = size of margin between classes
%          params.R = latent subspace dimensionality
%          params.Hsigma2 = variance of latent subspace
%
% Ouputs
% params = a structure of updated hyperparameters 
%          params.Lambda = inferred projection matrix hyperpriors (.theta .kappa)
%          params.A = inferred projection matrix distribution (.mu .sig .sig_diag)
%          params.eta = inferred weight hyperpiors (.theta .kappa)
%          params.bw = inferred bias and weight distributions (.mu .sig)
%
% Paul Gardner, Sheffield University 2019

% check labels are all [-1 +1]
labs = cellfun(@unique,y,'Uni',0);
flag = cellfun(@(c) c~=[-1 1]',labs,'Uni',0);
if any([flag{:}]~=0)
    error('Labels must be -1 or 1')
end

% sizes of domains
T = length(K); % no. of tasks
N = cellfun(@length,K,'Uni',0); % no. of points (K in R^NxN)

% unpack parameters
R = params.R; % dimension size of H (i.e. dimensionality reduction)
Hsig2 = params.Hsigma2; % prior variance of H

% initalise dimensionality reduction part

% approximate posterior of gamma shape parameters (kappa) and inialise gamma scale
% parameters (theta)
Lambda.kappa = cellfun(@(c) (params.lambda.kappa + 0.5)*ones(c,R),N,'Uni',0); % A hyperparameters
Lambda.theta = cellfun(@(c) params.lambda.theta*ones(c),N,'Uni',0); % A hyperparameters

% intialise A and H
A.mu = cellfun(@(c) randn(c,R),N,'Uni',0); % initalise A mu as random matrix
A.sig_diag = cellfun(@(c) ones(c,R),N,'Uni',0); % initalise diag of A sigma as unit variance
A.sig = cellfun(@(c) repmat(eye(c),[1 1 R]),N,'Uni',0); % initalise A sigma as unit variance
H.mu = cellfun(@(c) randn(R,c),N,'Uni',0); % initalise H mu as random matrix
H.sig = cellfun(@(c) eye(R),cell(1,T),'Uni',0); % initalise H sigma as unit variance

% initalise classifier part

% approximate posterior of gamma shape parameters (kappa) and inialise gamma scale
% parameters (theta)
gam.kappa = params.gamma.kappa + 0.5; % b hyperparameters
eta.kappa = (params.eta.kappa + 0.5)*ones(R,1); % w hyperparameters
gam.theta = params.gamma.theta; % b hyperparameters
eta.theta = params.eta.theta*ones(R,1); % w hyperparameters

% weights and bias
bw.mu = [0; randn(R,1)]; % initalise mean weights and bias
bw.sig = eye(R+1); % initalise weigths and bias variance

% intialise discriminative function
f.mu = cellfun(@(c1,c2) (abs(randn(c1,1)) + params.margin).*sign(c2),N,y,'Uni',0); % initalise function (make sure either side of margin)
f.sig = cellfun(@(c) ones(c,1),N,'Uni',0); % initalise unit variance

% get cross product K*K'
KKt = cellfun(@(c) c*c',K,'Uni',0);

% margins
labs = cellfun(@(c) [c<0, c>0],y,'Uni',0); % find -1 and +1 labels
margin_plus = cellfun(@(c) repmat([-1e40,+params.margin],c,1),N,'Uni',0); % -1e40 (if +1) +margin (if -1)
margin_minus = cellfun(@(c) repmat([-params.margin,+1e40],c,1),N,'Uni',0); % -margin (if +1) +1e40 (if -1)
lower_margin = cellfun(@(c1,c2) c1(c2),margin_plus,labs,'Uni',0); % lower marign
upper_margin = cellfun(@(c1,c2) c1(c2),margin_minus,labs,'Uni',0); % upper margin

% variation inference steps
for i = 1:params.iter
    
    % update dimensionality reduction part
    
    % update lamda theta - hyperparameters of projection prior
    Lambda.theta = cellfun(@(c1,c2) 1./(1./params.lambda.theta + 0.5*(c1.^2 + c2)),A.mu,A.sig_diag,'Uni',0);
    
    % update A - projection matrix
    for t = 1:T
        for s = 1:R
            % update A covariance
            A.sig{t}(:,:,s) = (diag(Lambda.kappa{t}(:,s).*Lambda.theta{t}(:,s)) + KKt{t}/Hsig2)\eye(N{t});
            A.sig_diag{t}(:,s) = diag(A.sig{t}(:,:,s)); % store diag
            % update A mean
            A.mu{t}(:,s) = A.sig{t}(:,:,s)*(K{t}*H.mu{t}(s,:)'/Hsig2);
        end
    end
    
    % update H - latent subspace
    H.sig = cellfun(@(c) (eye(R)/Hsig2 + bw.mu(2:R+1)*bw.mu(2:R+1)' + ...
        bw.sig(2:R+1,2:R+1))\eye(R),cell(1,T),'Uni',0);
    H.mu = cellfun(@(c1,c2,c3,c4,c5) c1*(c2'*c3/Hsig2 + bw.mu(2:R+1)*c4' ...
        - repmat(bw.mu(2:R+1)*bw.mu(1) + bw.sig(2:R+1,1),1,c5)),H.sig,A.mu,K,f.mu,N,'Uni',0);
    
    % update classifier part
    
    % update gamma theta - hyperparameter of bias prior
    gam.theta = 1/(1/params.gamma.theta + 0.5*(bw.mu(1)^2 + bw.sig(1,1)));
    % update eta theta - hyperparameters of weight prior
    eta.theta = 1./(1/params.eta.theta + 0.5*(bw.mu(2:R+1).^2 + diag(bw.sig(2:R+1,2:R+1))));
    
    % update bias and weights
    % bias and weight covariance
    bw.sig = [gam.kappa*gam.theta,zeros(1,R);zeros(R,1),diag(eta.kappa.*eta.theta)]; % non-task dependant
    bwsig = cellfun(@(c1,c2,c3) [c1, sum(c2,2)';...
        sum(c2,2), c2*c2' + c1*c3],N,H.mu,H.sig,'Uni',0); % create addition for each task
    bw.sig = (bw.sig + sum(cat(3,bwsig{:}),3))\eye(R+1,R+1); % sum all tasks and find inverse
    % bias and weight mean
    bwmu = cellfun(@(c1,c2,c3) [ones(1,c1);c2]*c3,N,H.mu,f.mu,'Uni',0); % create addition for each task
    bw.mu = bw.sig*sum(cat(2,bwmu{:}),2); % mean bias and weight
    
    % update f
    q_f_mu = cellfun(@(c1,c2,c3) [ones(1,c1); c2]'*bw.mu,N,H.mu,'Uni',0); % posterior update of f mu
    % truncated normal
    alpha_tn = cellfun(@(c1,c2) c1 - c2,lower_margin,q_f_mu,'Uni',0); % alpha for truncated normal
    beta_tn = cellfun(@(c1,c2) c1 - c2,upper_margin,q_f_mu,'Uni',0); % beta for truncated normal
    Z = cellfun(@(c1,c2) normcdf(c1) - normcdf(c2),beta_tn,alpha_tn,'Uni',0); % normalising constant
    Z = cellfun(@(c) c + (c==0) ,Z,'Uni',0); % normalising constant
    % f mean and variance
    f.mu = cellfun(@(c1,c2,c3,c4) c1 + (normpdf(c2)-normpdf(c3))./c4,q_f_mu,alpha_tn,beta_tn,Z,'Uni',0); % f mean through truncated normal
    f.sig = cellfun(@(c1,c2,c3) 1 + (c1.*normpdf(c1) - c2.*normpdf(c2))./c3 - ((normpdf(c1)-normpdf(c2))./c3).^2,alpha_tn,beta_tn,Z,'Uni',0); % f sigma through truncated normal
    
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