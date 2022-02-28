%% This script runs the Gnat repair problem from 
% Overcoming the problem of repair in structural health monitoring: 
% Metric-informed transfer learning
%
% Paul Gardner, University of Sheffield 2021
%
% note results might vary slighlty from the paper due to different random
% seeds in the training-testing split

clear all
close all
clc

%% Set transfer learning paths

addpath('../util')
addpath('../kernels')
addpath('../classifiers')
addpath('../models')

%% Load data

load('..\data\gnat_repair')

%% Training and testing splits

N = size(Xs,1);
ind_rand = randperm(N); % randomise intergers
ntr = sort(ind_rand(1:500)); % training data indices
ntst = sort(ind_rand(501:end)); % testing data indices

% source training and testing
Xs_tr = Xs(ntr,:);
Ys_tr = Ys(ntr,:);
Xs_tst = Xs(ntst,:);
Ys_tst = Ys(ntst,:);

% target training and testing
Xt_tr = Xt(ntr,:);
Yt_tr = Yt(ntr,:);
Xt_tst = Xt(ntst,:);
Yt_tst = Yt(ntst,:);

%% Normalise

% means and standard deviations
xs_mu = mean(Xs_tr);
xs_std = std(Xs_tr);
xt_mu = mean(Xt_tr);
xt_std = std(Xt_tr);

% normalise
Xs_tr = (Xs_tr - xs_mu)./xs_std;
Xt_tr = (Xt_tr - xt_mu)./xt_std;
Xs_tst = (Xs_tst - xs_mu)./xs_std;
Xt_tst = (Xt_tst - xt_mu)./xt_std;

%% M-JDA

% Domain adaptation parameters
kern = @kernelRBF; % kernel function
hyp = nan; % hyperparameters of kernel
mu = 0.1; % regularisation parameter
classifier = @classifierKNN_cv; % classifier
k = 2; % transfer space dimension
iter = 1; % number of psuedo-labelling iteration 

% Metric-informed labelling parameters
nMC = 10000; % number of monte carlo simulations for threshold
ne = 1000; % no. of ensembles
cb = 99; % confidence bound
nf = 8; % no. of features (sqrt(N) is guiding heuristic)

%% M-JDA

[Zs_tr,Zt_tr,Ytp,W,cls,~,mmd] = mjda(Xs_tr,Ys_tr,Xt_tr,...
    kern,hyp,...
    mu,k,classifier,iter,...
    nf,ne);

% transform test datasets using weights
Zs_tst = domainAdaptationTransform(Xs_tst,Xs_tr,Xt_tr,...
    W,kern,hyp);
Zt_tst = domainAdaptationTransform(Xt_tst,Xs_tr,Xt_tr,...
    W,kern,hyp);

%% Classification using a KNN on the M-JDA transformed space

% training (source) data
Ysp_tr = classifier(Zs_tr,Ys_tr,Zs_tr,cls);
Ysp_tst = classifier(Zs_tr,Ys_tr,Zs_tst,cls);

% metrics
% training (source)
acc_s_tr = accuracy(Ysp_tr,Ys_tr);
[f1_s_tr,f1_k_s_tr] = f1score(Ysp_tr,Ys_tr);
% testing (source)
acc_s_tst = accuracy(Ysp_tst,Ys_tst);
[f1_s_tst,f1_k_s_tst] = f1score(Ysp_tst,Ys_tst);

% target data
Ytp_tr = classifier(Zs_tr,Ys_tr,Zt_tr,cls);
Ytp_tst = classifier(Zs_tr,Ys,Zt_tst,cls);

% metrics
% training (source)
acc_t_tr = accuracy(Ytp_tr,Yt_tr);
[f1_t_tr,f1_k_t_tr] = f1score(Ytp_tr,Yt_tr);
% testing (source)
acc_t_tst = accuracy(Ytp_tst,Yt_tst);
[f1_t_tst,f1_k_t_tst] = f1score(Ytp_tst,Yt_tst);

%% Visualise M-JDA mapping (with true labels used to aid visualisation of the mapping)

figure
hold on
gscatter(Zs_tr(:,1),Zs_tr(:,2),Ys_tr,[],'.',7)
gscatter(Zt_tr(:,1),Zt_tr(:,2),Yt_tr,[],'o',5)
gscatter(Zs_tr(:,1),Zs_tr(:,2),Ys_tr,[],'.',7)
xlabel('1st transfer component')
ylabel('2nd transfer component')
title('Training data mapping')

figure
hold on
gscatter(Zs_tst(:,1),Zs_tst(:,2),Ys_tst,[],'.',7)
gscatter(Zt_tst(:,1),Zt_tst(:,2),Yt_tst,[],'o',5)
gscatter(Zs_tst(:,1),Zs_tst(:,2),Ys_tst,[],'.',7)
xlabel('1st transfer component')
ylabel('2nd transfer component')
title('Testing data mapping')
