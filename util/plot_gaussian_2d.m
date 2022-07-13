function [X,Y] = plot_gaussian_2d(mu,sigma,npts,stds)
% plots k-Gaussian distributions given mean and covariances in 2D
%
% Inputs
% mu = k-means (k*d)
% sigma = k-covariances (d*d*k)
% npts = no. of points on plot of covariances
% stds = no. of standard deviations on plot
%
% Outputs
% X = x-coordinate of covariances (k*npts)
% Y = y-coordinate of covariances (k*npts)
% also makes plot
%
% Paul Gardner, University of Sheffield 2022

if nargin<3
    npts = 50; % default no. of points
end

if nargin<4
    stds = 2; % default no. of standard deviations
end

k = size(mu,1); % number of components

theta = 2*pi./npts.*(0:npts); % points in circle

hold on % figure already assumed
plot(mu(:,1),mu(:,2),'k+')
X = nan(k,npts+1);
Y = nan(k,npts+1);
for i = 1:k
    % eigen-scaling of circle
    [eigvec,eigval] = eig(sigma(:,:,i));
    alpha = atan(eigvec(2,1)/eigvec(1,1));
    
    xy = stds.*sqrt(diag(eigval)).*[cos(theta); sin(theta)]; % radius of circle
    ellip = [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)]*xy; % rotate circle
    
    X(i,:) = mu(i,1)+ellip(1,:); % add mean offset x
    Y(i,:) = mu(i,2)+ellip(2,:); % add mean offset y
    
    plot(X(i,:),Y(i,:),'k-')
end
hold off
legend(arrayfun(@num2str,1:k,'UniformOutput',0))

end