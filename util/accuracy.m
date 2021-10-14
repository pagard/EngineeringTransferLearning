function acc = accuracy(yp,yt)
% accuracy of classifier
%
% Inputs
% yp = predicted labels
% yt = true labels
%
% Output
% acc = accuracy

acc = 100*(length(find(yp==yt))/length(yt));