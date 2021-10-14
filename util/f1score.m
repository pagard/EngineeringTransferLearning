function [f1,f1_c] = f1score(yt,yp)
% marco f1 score of classifier
%
% Inputs
% yp = predicted labels
% yt = true labels
%
% Outputs
% f1_c = f1 per class
% f1 = f1 score

classes = unique(yt); % classes
C = length(classes); % no. of classes

cp = repmat(yt,1,C) == classes'; % predicted binary class matrix
ct = repmat(yp,1,C) == classes'; % true binary class matrix

TP = sum(cp == 1 & ct == 1,1); % true positive
FN = sum(cp == 0 & ct == 1,1); % false negative
FP = sum(cp == 1 & ct == 0,1); % false positive

pre = TP./(TP+FP); % precision
rec = TP./(TP+FN); % recall

f1_c = 2.*(pre.*rec)./(pre+rec); % f1 score per class
f1 = mean(f1_c); % marco f1 score
