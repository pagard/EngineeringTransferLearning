function [Ytp,Ysp] = metricInformedLabels(Xs,Ys,Xt,nf,ne,cb,nMC)
% Metric-informed psuedo-labels using MSD distances
%
% Inputs
% Xs = source features (ns*d)
% Ys = source labels (ns*1)
% Xt = target features (nt*d)
% nf = no. of features (sqrt(N) heuristic)
% ne = no. of ensembles
% cb = threshold confidence bound
% nMC = no. of monte carlo samples for threshold
%
% Outputs
% Ytp = target psuedo-labels
% Ysp = source psuedo-labels

if nargin<6, cb = 99; end
if nargin<7, nMC = 10000; end

classes = unique(Ys); % classes
C = length(classes); % no. of classes

ns = size(Xs,1); % length of source data
nt = size(Xt,1); % length of target data

X = [Xs;Xt]; % combine data

if nargin>=5
    dm = nan(ns+nt,ne); % distances per model in ensemble
end
dc = nan(ns+nt,length(classes)); % MSD distances per class
dc_norm = dc; % normalised MSD distances per class

for i = 1:C
    
    Xs_c = Xs(Ys == classes(i),:); % source data for class i
    df = size(Xs_c,2); % dimension of feature
    
    if df == nf
        
        % MSD model's sample mean and covariance
        mu = mean(Xs_c);
        sig = cov(Xs_c);
        
        % MSD
        try
            dc(:,i) = MSD(X,mu,sig);
        catch
            dc(:,i) = MSD(X,mu,sig+eye(size(sig))*1e-12);
            warning('Sig not invertible added a jitter');
        end
        
    else
        % MSD ensembles
        for j = 1:ne
            
            % feature bag
            feat_ind = sort(randperm(df,nf)); % samples features
            fb_s_c = Xs_c(:,feat_ind); % feature bag training data
            fb = X(:,feat_ind); % feature bag complete dataset
            
            % MSD model's sample mean and covariance
            mu = mean(fb_s_c);
            sig = cov(fb_s_c);
            
            % MSD
            try
                dm(:,j) = MSD(fb,mu,sig);
            catch
                dm(:,j) = MSD(fb,mu,sig+eye(size(sig))*1e-12);
                warning('Sig not invertible added a jitter');
            end
            
        end
        dc(:,i) = mean(dm,2); % ensemble averaged feature
    end
    
    % Monte Carlo threshold for class
    thres = nan(nMC,1);
    for j = 1:nMC
        x_zeta = randn(size(Xs_c,1),nf);
        thres(j) = max(MSD(x_zeta,zeros(1,size(x_zeta,2)),eye(size(x_zeta,2))));
    end
    thres = sort(thres);
    thres = thres(floor(nMC*cb/100));
    
    % normalise by threshold
    dc_norm(:,i) = dc(:,i)/thres;
    dc_norm(dc(:,i)<thres,i) = 0; % below threshold set to 0
end

% Psuedo-labels
[~,ind] = min(dc_norm,[],2); % min distance per class
Ypsuedo = classes(ind); % psuedo labels based on minimum distances
Ysp = Ypsuedo(1:ns); % source psuedo-labels
Ytp = Ypsuedo(ns+1:ns+nt); % target psuedo-labels

    function d = MSD(X,Xmu,Xcov)
        % mahalanobis distance
        R = chol(Xcov);
        res = X - repmat(Xmu,size(X,1),1);
        d = diag(res*(R\(R'\res')));
    end
end

