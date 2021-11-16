%% This script runs the multi-class problem from
% On the application of kernelised Bayesian transfer learning to
% population-based structural health monitoring
%
% Paul Gardner, Sheffield University 2021
%
% note results might vary slighlty from the paper due to different random
% seeds in the random initialisation of KBTL

clear all
close all
clc

%% Set transfer learning paths

addpath('../util')
addpath('../kernels')
addpath('../models/kbtl')

%% Load data

load('..\data\kbtl_demo_multiclass_data')

%% KBTL hyperprior parameters

% shape and scale hyperparameters of gamma prior for projection matrices
params.lambda.kappa = 1e-3;
params.lambda.theta = 1e-3;

% shape and scale hyperparameters of gamma prior for bias
params.gamma.kappa = 1e-3;
params.gamma.theta = 1e-3;

% shape and scale hyperparameters of gamma prior for weights
params.eta.kappa = 1e-3;
params.eta.theta = 1e-3;

% no. of iterations
params.iter = 5000; % this has been reduced from the paper

% margin
params.margin = 1;

% latent subspace dimensionality
params.R = 2;

% variance of latent subspace
params.Hsigma2 = 6^2;

%% Train KBTL

T = length(X); % no. of tasks

% get kernel embeddings
Ktrain = cell(1,T); % training kernels
khyp = nan(T,1); % kernel hyperparameters
for t = 1:T
    [Ktrain{t},khyp(t)] = kernelRBF(nan,X{t},X{t}); % kernel embedding (and kernel hyperparameters using median heuristic)
end

hyp = kbtl_train(Ktrain,Y,params); % train kbtl
pred_train = kbtl_test(Ktrain,hyp,Y); % predict training data

%% Plot training predictions

symbs = {'x','s','p','*','d','^','.'}; % markers for plot
clr = [ 0.1  0.6 0.45;
       0.85  0.4 0.01;
       0.45 0.45  0.7;
        0.9  0.2 0.55]; % class colours

kbtl_2d_plot_mc(pred_train,Y,hyp,clr,symbs) % plot latent subspace

pred_train.acc

%% KBTL Prediction

% get kernel embeddings
Ktest = cell(1,T); % test kernels
for t = 1:T
    Ktest{t} = kernelRBF(khyp(t),X{t},Xtest{t}); % kernel embedding (and kernel hyperparameters using median heuristic)
end

pred_test = kbtl_test(Ktest,hyp,Ytest); % predict testing data

%% Plot testing predictions

kbtl_2d_plot_mc(pred_test,Ytest,hyp,clr,symbs) % plot latent subspace

% accuracy
pred_test.acc

%%  KBTL multi-class 2D plotting function

function kbtl_2d_plot_mc(pred,y,hyp,clr,symbs)

nC = size(hyp.bw.mu,2); % number of classes

% Bounds of latent subspace
max_bounds = cellfun(@(c) max(c,[],2)',pred.H.mu,'Uni',0);
max_bounds = max(cat(1,max_bounds{:})) + 0.2*max(cat(1,max_bounds{:})).*sign(max(cat(1,max_bounds{:})));
min_bounds = cellfun(@(c) min(c,[],2)',pred.H.mu,'Uni',0);
min_bounds = min(cat(1,min_bounds{:})) - 0.2*min(cat(1,min_bounds{:})).*sign(min(cat(1,min_bounds{:})));

% Calculate linear boundary in subspace
xx = linspace(min_bounds(1),max_bounds(1),1000)';
yy = nan(size(xx,1),nC);
for i = 1:nC
    yy(:,i) = (-hyp.bw.mu(1,i) - hyp.bw.mu(2,i).*xx)/hyp.bw.mu(3,i);
end

% function variance
fsig = nan(size(xx,1),nC);
for i = 1:nC
    fsig(:,i) = 1 + diag([ones(1,size(xx,1)); [xx';yy(:,i)']]'*hyp.bw.sig(:,:,i)*[ones(1,size(xx,1)); [xx';yy(:,i)']]);
end

T = length(y); % no. of tasks

figure('position',[500,500,900,400])
subplot(1,2,1)
hold on
for i = 1:nC
    plot(nan,nan,'.','color',clr(i,:),'markersize',20)
end
for i = 1:nC
    plot(xx,yy(:,i),'-','color',clr(i,:),'linewidth',2)
    plot(xx,yy(:,i)+3*sqrt(fsig(:,1)),'--','color',clr(i,:),'linewidth',1)
    plot(xx,yy(:,i)-3*sqrt(fsig(:,1)),'--','color',clr(i,:),'linewidth',1)
end

for i = 1:T
    if i ~= T
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),pred.ymap{i},clr(unique(pred.ymap{i}+1),:),symbs{i},3);
    else
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),pred.ymap{i},'k','o',10);
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),pred.ymap{i},clr(unique(pred.ymap{i}+1),:),symbs{i},30);
    end
end
legend(arrayfun(@(a) cellstr(num2str(a)),unique(cat(1,y{:}))),'Orientation','horizontal','fontsize',12,'Position',[0.36 0.945 0.3 0.05]);
set(gca,'fontsize',12)
xlabel('H[:,1]')
ylabel('H[:,2]')
xlim([min_bounds(1) max_bounds(1)])
ylim([min_bounds(2) max_bounds(2)])

subplot(1,2,2)
hold on
for i = 1:nC
    plot(nan,nan,'.','color',clr(i,:),'markersize',20)
end
for i = 1:nC
    plot(xx,yy(:,i),'-','color',clr(i,:),'linewidth',2)
    plot(xx,yy(:,i)+3*sqrt(fsig(:,1)),'--','color',clr(i,:),'linewidth',1)
    plot(xx,yy(:,i)-3*sqrt(fsig(:,1)),'--','color',clr(i,:),'linewidth',1)
end

for i = 1:T
    if i ~= T
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),y{i},clr(unique(y{i}+1),:),symbs{i},3);
    else
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),y{i},'k','o',10);
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),y{i},clr(unique(y{i}+1),:),symbs{i},30);
    end
end
legend('off')
set(gca,'fontsize',12)
xlabel('H[:,1]')
ylabel('H[:,2]')
xlim([min_bounds(1) max_bounds(1)])
ylim([min_bounds(2) max_bounds(2)])

end