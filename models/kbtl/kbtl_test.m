function pred = kbtl_test(K,params,y)
% Multi-class kernelised bayesian transfer learning - testing
%
% Inputs
% K = cell of kernels from each domain
% Y = cell of class numeric labels for each domain (these get converted to
% [-1 +1] N x L matrix) or cell of labels in N x L matrix form
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
%          params.Lambda = inferred projection matrix hyperpriors (.theta .kappa)
%          params.A = inferred projection matrix distribution (.mu .sig .sig_diag)
%          params.eta = inferred weight hyperpiors (.theta .kappa)
%          params.bw = inferred bias and weight distributions (.mu .sig)
%
% Outputs
% pred = a structure containing the predictive variables
%       pred.H.mu = mean predictive latent space
%       pred.f = mean predictive function (.mu .sig)
%       pred.py = probability of +1 class
%       pred.ymap = MAP estimate of class label
% if y is known
%       pred.acc = prediction accuracy
%       pred.f1 = prediction f1 score
%
% Paul Gardner, Sheffield University 2019

% size of test domains
T = length(K); % no. of tasks
N = cellfun(@(c) size(c,2),K,'Uni',0); % no. of points (K in R^NxN)

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

L = size(params.bw.mu,2); % number of classes in training

% check if no test data
no_pts = cellfun(@(c) any(c==0),N,'Uni',1);
if any(no_pts ~=0)
    tt = find(no_pts == 0);
    for t = 1:tt
        K{t} = nan(N{t});
    end
end

% predict dimensionality reduction subspace
pred.H.mu = cellfun(@(c1,c2) c1'*c2,params.A.mu,K,'Uni',0);

% predictive function
pred.f.mu = cellfun(@(c1,c2) [ones(1,c1); c2]'*params.bw.mu,N,pred.H.mu,'Uni',0);
for ll = 1:L
    pred.fsig(:,ll) = cellfun(@(c1,c2) 1 + diag([ones(1,c1); c2]'*params.bw.sig(:,:,ll)*[ones(1,c1); c2]),N,pred.H.mu,'Uni',0);
end
for t = 1:T
    pred.f.sig{t} = cat(2,pred.fsig{t,:});
end

% probability of y = +1
prob_plus = cellfun(@(c1,c2) normcdf((c1 - params.margin)./c2),pred.f.mu,pred.f.sig,'Uni',0);
prob_minus = cellfun(@(c1,c2) 1 - normcdf((c1 - params.margin)./c2),pred.f.mu,pred.f.sig,'Uni',0);
pred.py = cellfun(@(c1,c2) c1./(c1+c2),prob_plus,prob_minus,'Uni',0);

% map estimate
for t = 1:T
    [~,yind] = max(pred.py{t},[],2);
    pred.ymap{t} = labs(yind);
end

% map estimate of labels
% map_minus = cellfun(@(c) c<0.5,pred.py,'Uni',0); % find p(y*)<0.5
% pred.ymap = cellfun(@(c1,c2) ones(c1,1)-2*c2,N,map_minus,'Uni',0); % map estimate labels

% performance metrics
if nargin == 3 && all(Ls==1)
    addpath('..\util')
    pred.acc = cellfun(@(c1,c2) 100*(length(find(c1==c2))./length(c2)),pred.ymap,y); % accuracy
    pred.f1 = cellfun(@(c1,c2) f1score(c1,c2),pred.ymap,y); % f1 macro
end

end