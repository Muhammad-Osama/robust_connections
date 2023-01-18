function df = degrees_of_freedom(Phi_mat, lambda_0, lambda_vec)
% Input
% Phi_mat: n x nBasis matrix
% lambda_0 : covariance parameter lambda0
% lambda_vec : d x 1 vector of covariance vector

n = size(Phi_mat, 1);

M = (Phi_mat * diag(lambda_vec) * Phi_mat' + lambda_0 * eye(n))\...
    (Phi_mat * diag(lambda_vec) * Phi_mat');

% degrees of freedom
df = trace(M);
