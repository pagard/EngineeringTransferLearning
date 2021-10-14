function [Zs,Zt,Ytp,W,cls,fscore,mmd] = bda(Xs,Ys,Xt,kern,hyp,mu,k,lambda,classifier,iter,mode,Yt)
% Balanced distribution adaptation
%
% Inputs
% Xs = source features (ns*d)
% Ys = source labels (ns*1)
% Xt = target features (nt*d)
% kern = function handle of kernel function or no function handle for X
% hyp = kernel hyperparameters
% mu = trade-off parameter
% k = no. of eigenvectors (dimension of embedding <= d)
% lambda = balance factor, between [0 1], e.g. 1 ignores the marginal
% classifer = function handle for classifer train & prediction
% iter = no. of iterations for psuedo-labelling
% mode = indicator if BDA or Weight-BDA (WBDA) is used. For WBDA mode = 1
% Yt = target labels (nt*1)
%
% Outputs
% Zs = k-dimension transformed source feature space
% Zt = k-dimension transformed target feature space
% Ytp = pseudo target class labels
% W = k-dimension embedding matrix (or adaptation matrix)
% cls = parameters of classifier
% mmd = mmd distance of transformed space

% Paul Gardner, Sheffield University 2021

ns = size(Xs,1); % length of source data
nt = size(Xt,1); % length of target data

X = [Xs;Xt]; % combine data
[n,d] = size(X); % length and dimension of total data set

if k>d; error('You ask for too many eigenvectors'); end

% calculate M0 - constants from MMD biased v-statistic
M0 = zeros(n) -1/(ns*nt);
M0(1:ns,1:ns) = 1/ns^2;
M0(ns+1:end,ns+1:end) = 1/nt^2;

% calculate H
H = eye(n) - ones(n)/(n); % centering matrix

% kernel embedding
if isa(kern,'function_handle')
    % Calculate kernel matrix
    K = kern(hyp,X,X);
else
    % No kernel embedding
    K = X;
    n = d;
end

% Initial psuedo-labels and classifier
[Ytp,cls] = classifier(Xs,Ys,Xt); 

fs_best = 0; % initalise best f score
failed = 0; % counter if weights are imaginary
fscore = nan(iter,1);

for i = 1:iter % some number till convergence...
    
    M = zeros(size(M0)); % initalise
    for c = unique(Ys)'
                
        cYs = find(c==Ys); % index's of Xs's in class c
        cYtp = find(c==Ytp); % index's of Xt's in class c
        
        % Mc
        nsc = length(cYs); % number of source inputs in the class
        ntc = length(cYtp); % number of target inputs in the class
        
        if mode == 1
            Py_s = nsc/ns; % prior probability of source class label
            Py_t = ntc/nt; % prior probability of target class label
        else
            % set prior probability to 1 when BDA
            Py_s = 1; 
            Py_t = 1;
        end
        
        % MMD or weight matrix
        M(cYs,cYs) = M(cYs,cYs) + Py_s/nsc^2;
        M(ns+cYtp,ns+cYtp) = M(ns+cYtp,ns+cYtp) + Py_t/ntc^2;
        M(cYs,ns+cYtp) = M(cYs,ns+cYtp) -sqrt(Py_s*Py_t)/(nsc*ntc);
        M(ns+cYtp,cYs) = M(ns+cYtp,cYs) -sqrt(Py_s*Py_t)/(nsc*ntc);
    end
    % balanced MMD matrix
    M = (1-lambda)*M0 + lambda*M;
            
    % k smallest eigenvectors
    [W,~] = eigs(mu*eye(n)+K'*M*K, K'*H*K,k,'sm');
    
    if isreal(W) % check solution is real
        Z = K*W; % transform k-components
        
        % extract transformed source and target data
        Zs = Z(1:ns,:);
        Zt = Z(ns+1:ns+nt,:);
        
        % Psuedo-labels and classifier
        [Ytp,cls] = classifier(Zs,Ys,Zt);
                
        % if target labels are used for validation
        if nargin==12
            % calculate f1 score
            fscore(i) = f1score(Ytp,Yt);
            fprintf('F Score %2.3f\n',fscore(i))
            
            % update best classifier and mapping
            if i == 1 || fscore(i) > fs_best
                cls_best = cls;
                W_best = W;
                Zs_best = Zs;
                Zt_best = Zt;
                fs_best = fscore(i);
                Ytp_best = Ytp;
            end
        end
        
    else
        % weights weren't real
        
        %fprintf('Iteration %2d not real weights \n',i)
        failed = failed + 1;  
    end
    
end

% outputs
if nargin == 12 && failed ~= iter
    W = W_best;
    Zs = Zs_best;
    Zt = Zt_best;
    cls = cls_best;
    Ytp = Ytp_best;
    fprintf('Best F Score %2.3f\n',fs_best)
elseif failed == iter
    fprintf('All iterations failed\n')
    Zs = nan;
    Zt = nan;
    Ytp = nan;
    cls = nan;
end

% mmd of transfer space
mmd = sum(diag(W'*K*M*K*W));

end