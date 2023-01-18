function [lambda0, lambda_vec] = covariance_parameter_spice(y, Phi_mat, theta_spice)
% INPUT 
% y: n x 1 vector of output
% Phi_mat : n X nBasis matrix
% theta_spice : d x 1 vector of SPICE estimate

%%
% Estimate covariance parameters \lambda0, \lambda1, ...., \lambda_n
%nos of data points
n = length(y);

%lambda_0
lambda0 = norm(y - Phi_mat * theta_spice, 2)/sqrt(n);

% lambda1, ..., lambda_d
lambda_vec = abs(theta_spice)./(sqrt(sum((Phi_mat.^2), 1))');
