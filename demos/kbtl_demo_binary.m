%% This script runs the binary problem from
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

load('..\data\kbtl_demo_binary_data')

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
params.iter = 500; % this has been reduced from the paper

% margin
params.margin = 0;

% latent subspace dimensionality
params.R = 2;

% variance of latent subspace
params.Hsigma2 = 0.25^2;

%% Train KBTL

T = length(X); % no. of tasks

% get kernel embeddings
Ktrain = cell(1,T); % training kernels
khyp = nan(T,1); % kernel hyperparameters
for t = 1:T
    [Ktrain{t},khyp(t)] = kernelRBF(nan,X{t},X{t}); % kernel embedding (and kernel hyperparameters using median heuristic)
end

hyp = kbtl_train_binary(Ktrain,Y,params); % train kbtl
pred_train = kbtl_test_binary(Ktrain,hyp,Y); % predict training data

%% Plot training predictions

symbs = {'x','s','p','*','d','^','.'}; % markers for plot
ud = 0.1*(1:7)'*[0.4 0.75 0.65]; % colours for undamaged
d = 0.1*(1:7)'*[0.98 0.55 0.4]; % colours for damaged

kbtl_2d_plot_binary(pred_train,Y,hyp,ud,d,symbs); % plot latent subspace (left predictive labels, right true labels)

% accuracy
pred_train.acc

%% KBTL Prediction

% get kernel embeddings
Ktest = cell(1,T); % test kernels
for t = 1:T
    Ktest{t} = kernelRBF(khyp(t),X{t},Xtest{t}); % kernel embedding (and kernel hyperparameters using median heuristic)
end

pred_test = kbtl_test_binary(Ktest,hyp,Ytest); % predict testing data

%% Plot testing predictions

kbtl_2d_plot_binary(pred_test,Ytest,hyp,ud,d,symbs); % plot latent subspace (left predictive labels, right true labels)

% accuracy
pred_test.acc

%% KBTL binary 2D plotting function

function kbtl_2d_plot_binary(pred,y,hyp,clr_1,clr_2,symbs)

% Bounds of latent subspace
max_bounds = cellfun(@(c) max(c,[],2)',pred.H.mu,'Uni',0);
max_bounds = max(cat(1,max_bounds{:})) + 0.2*max(cat(1,max_bounds{:})).*sign(max(cat(1,max_bounds{:})));
min_bounds = cellfun(@(c) min(c,[],2)',pred.H.mu,'Uni',0);
min_bounds = min(cat(1,min_bounds{:})) - 0.2*min(cat(1,min_bounds{:})).*sign(min(cat(1,min_bounds{:})));

% Calculate linear boundary in latent subspace
xx = (min_bounds(1):0.1:max_bounds(1))';
yy = (-hyp.bw.mu(1) - hyp.bw.mu(2).*xx)/hyp.bw.mu(3);

% function variance
fsig = 1 + diag([ones(1,size(xx,1)); [xx';yy']]'*hyp.bw.sig*[ones(1,size(xx,1)); [xx';yy']]);

T = length(y); % no. of tasks

figure('position',[500,500,900,400])
subplot(1,2,1)
hold on
plot(nan,nan,'.','color',clr_1(end,:),'markersize',20)
plot(nan,nan,'.','color',clr_2(end,:),'markersize',20)
plot(xx,yy,'k-','linewidth',2)
plot(xx,yy+3*sqrt(fsig),'--','color','k','linewidth',1)
plot(xx,yy-3*sqrt(fsig),'--','color','k','linewidth',1)
for i = 1:T
    if i ~= T
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),pred.ymap{i},[clr_1(i,:);clr_2(i,:)],symbs{i},5);
    else
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),pred.ymap{i},'k','o',10);
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),pred.ymap{i},[clr_1(i,:);clr_2(i,:)],symbs{i},30);
    end
end
legend({'0','1'},'Orientation','horizontal','fontsize',12,'Position',[0.5-0.1/2+0.01 0.945 0.1 0.05]);
set(gca,'fontsize',12)
xlabel('H[:,1]')
ylabel('H[:,2]')
xlim([min_bounds(1) max_bounds(1)])
ylim([min_bounds(2) max_bounds(2)])

subplot(1,2,2)
hold on
plot(xx,yy,'k-','linewidth',2)
plot(xx,yy+3*sqrt(fsig),'--','color','k','linewidth',1)
plot(xx,yy-3*sqrt(fsig),'--','color','k','linewidth',1)
for i = 1:T
    if i ~= T
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),y{i},[clr_1(i,:);clr_2(i,:)],symbs{i},5);
    else
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),y{i},'k','o',10);
        gscatter(pred.H.mu{i}(1,:),pred.H.mu{i}(2,:),y{i},[clr_1(i,:);clr_2(i,:)],symbs{i},30);
    end
end
legend('off')
set(gca,'fontsize',12)
xlabel('H[:,1]')
ylabel('H[:,2]')
xlim([min_bounds(1) max_bounds(1)])
ylim([min_bounds(2) max_bounds(2)])

end