%% This script runs the analysis for the four common subgraphs from
% A population-based SHM methodology for heterogeneous structures:
% transferring damage localisation knowledge between different aircraft
% wings
%
% Paul Gardner, University of Sheffield 2022

clear all
close all
clc

%% Set transfer learning paths

addpath('../util')
addpath('../kernels')
addpath('../classifiers')
addpath('../models')

%% Load data - datasets have been preprocessed using the steps in the paper
% X denotes PCA subspace features
% Tgnat_norm denotes normalised Gnat transmissibilities
% Tpiper_norm denotes normalised Piper transmissibilities
% P denotes a subspace basis
% Y denotes labels

load('..\data\gnat_piper_preprocessed_features.mat')
load('..\data\gnat_piper_preprocessed_labels.mat')
load('..\data\gnat_piper_preprocessed_pca_bases.mat')
load('..\data\gnat_piper_preprocessed_transmissibilities_tr.mat')
load('..\data\gnat_piper_preprocessed_transmissibilities_tst.mat')

%% Statistical Distances

d_sdm = 10; % dimension of PCA features from subspace disagreement measure
% (see paper)

kern = @kernelRBF; % RBF kernel for MMD distances

ROD = nan(4,1); % rank of domain
mmd = nan(4,1); % marginal maximum mean discrepancy
mmd_c = nan(4,1); % (approx.) joint maximum mean discrepancy

for i = 1:4
    
    ROD(i) = rankOfDomain(Ps{i},Pt,... % subspace basis
        Tgnat_norm_tr{i},Tpiper_norm_tr,... % datasets (before the linear transform)
        d_sdm); % dimensions from SDM
    
    [mmd(i),mmd_c(i)] = MMD(Xs_tr{i},Xt_tr,...% datasets
        kern,nan,... % kernel funciton and hyperparameters (nan used median heuristic)
        Ys_tr{i},Yt_tr); % labels for joint mmd
end

% plot bar chart of distance metrics
figure('position',[500,500,1000,300])
subplot(1,3,1)
bar(ROD)
xlabel('Maximum Common Subgraph')
ylabel('Rank of domain')
xticklabels({'(a)','(b)','(c)','(d)'});

subplot(1,3,2)
bar(mmd)
xlabel('Maximum Common Subgraph')
ylabel('Marginal MMD')
xticklabels({'(a)','(b)','(c)','(d)'});

subplot(1,3,3)
bar(mmd_c)
xlabel('Maximum Common Subgraph')
ylabel('Joint MMD')
xticklabels({'(a)','(b)','(c)','(d)'});

%% Knowledge transfer using BDA

classifier = @classifierKNN_cv; % classifier function - cross validated KNN

% BDA parameters
opts.lambda = 1; % balance factor (1 means only conditionals minimised)
opts.kern = @kernelRBF; % kernel function
opts.hyp = nan; % nan uses median heuristic when using kernelRBF
opts.mu = 0.1; % regularisation parameter
opts.k = 2; % dimension of transfer component space
opts.classifier = classifier; % classifier

% preallocate
bda_Ysp_tr = cell(4,1); % BDA labels
bda_Ysp_tst = cell(4,1);
bda_Ytp_tr = cell(4,1);
bda_Ytp_tst = cell(4,1);

acc_s_tr_bda = nan(4,1); % accuracies and F1 scores
acc_s_tst_bda = nan(4,1);
f1_s_tr_bda = nan(4,1);
f1_k_s_tr_bda = nan(4,5);
f1_s_tst_bda = nan(4,1);
f1_k_s_tst_bda = nan(4,5);
acc_t_tr_bda = nan(4,1);
acc_t_tst_bda = nan(4,1);
f1_t_tr_bda = nan(4,1);
f1_k_t_tr_bda = nan(4,5);
f1_t_tst_bda = nan(4,1);
f1_k_t_tst_bda = nan(4,5);

cmap = [0.40    0.75    0.65
    0.9900    0.55    0.38
    0.55    0.63    0.80
    0.91    0.54    0.76
    0.65    0.85    0.33];

figure('position',[500 500 1500 300])
for i = 1:4
    
    % BDA
    
    % train bda mapping 
    [bda_Zs_tr,bda_Zt_tr,~,bda_W,cls] = bda(Xs_tr{i},Ys_tr{i},Xt_tr,...
        opts.kern,opts.hyp,opts.mu,opts.k,opts.lambda,opts.classifier,5,1);
    
    % transform test data
    bda_Zs_tst = domainAdaptationTransform(Xs_tst{i},Xs_tr{i},Xt_tr,...
        bda_W,opts.kern,opts.hyp);
    bda_Zt_tst = domainAdaptationTransform(Xt_tst,Xs_tr{i},Xt_tr,...
        bda_W,opts.kern,opts.hyp);
    
    % plot transfer components against true labels
    subplot(1,4,i)
    hold on
    gscatter(bda_Zs_tst(:,1),bda_Zs_tst(:,2),Ys_tst{i},cmap + [0 0.05 0.05 ],'.',12)
    gscatter(bda_Zt_tr(:,1),bda_Zt_tr(:,2),Yt_tr,cmap - [0 0.05 0.05 ],'o',10)
    gscatter(bda_Zt_tst(:,1),bda_Zt_tst(:,2),Yt_tst,cmap - [0 0.05 0.05 ],'o',10)
    gscatter(bda_Zs_tr(:,1),bda_Zs_tr(:,2),Ys_tr{i},cmap + [0 0.05 0.05 ],'.',12)
    xlabel('Z_1')
    ylabel('Z_2')  
    
    % classification and metrics
    
    % training (source) data
    bda_Ysp_tr{i} = classifier(bda_Zs_tr,Ys_tr{i},bda_Zs_tr,cls);
    bda_Ysp_tst{i} = classifier(bda_Zs_tr,Ys_tr{i},bda_Zs_tst,cls);
    
    % metrics
    % training (source)
    acc_s_tr_bda(i) = accuracy(bda_Ysp_tr{i},Ys_tr{i});
    [f1_s_tr_bda(i),f1_k_s_tr_bda(i,:)] = f1score(bda_Ysp_tr{i},Ys_tr{i});
    % testing (source)
    acc_s_tst_bda(i) = accuracy(bda_Ysp_tst{i},Ys_tst{i});
    [f1_s_tst_bda(i),f1_k_s_tst_bda(i,:)] = f1score(bda_Ysp_tst{i},Ys_tst{i});
    
    % target data
    bda_Ytp_tr{i} = classifier(bda_Zs_tr,Ys_tr{i},bda_Zt_tr,cls);
    bda_Ytp_tst{i} = classifier(bda_Zs_tr,Ys_tst{i},bda_Zt_tst,cls);
    
    % metrics
    % training (source)
    acc_t_tr_bda(i) = accuracy(bda_Ytp_tr{i},Yt_tr);
    [f1_t_tr_bda(i),f1_k_t_tr_bda(i,:)] = f1score(bda_Ytp_tr{i},Yt_tr);
    % testing (source)
    acc_t_tst_bda(i) = accuracy(bda_Ytp_tst{i},Yt_tst);
    [f1_t_tst_bda(i),f1_k_t_tst_bda(i,:)] = f1score(bda_Ytp_tst{i},Yt_tst);
    
end

% plot testing f1 score
figure
bar(f1_t_tst_bda)
xlabel('Maximum Common Subgraph')
ylabel('F1 Score')
xticklabels({'(a)','(b)','(c)','(d)'});
