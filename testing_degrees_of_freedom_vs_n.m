seed = 0;
rng(seed);
%%
% Generate data
n = 5;
x = linspace(0, 20, n)';
se = 0.25;
y = sin(2 * pi./20 .* x) + se .* randn(n, 1);
figure;
scatter(x, y);
xlabel('$x$','interpreter','Latex')
ylabel('$y$','interpreter','Latex');

%% 
%Fit SPICE
%Define basis function parameters
basis_support = 1.5;
nBasis = 10;
mn = 0; mx = 20; 
Phi_mat = zeros(n, nBasis); % n x d matrix of basis
for i = 1:n
    [Phi_mat(i, :), ~] = func_phi_bsplinebasis(x(i), mn, mx, nBasis, basis_support);
end
%Compute SPICE predictor
U = 1; % constant mean
L = 10; % nos. of itereations
w_hat_spice = compute_spicepredictor(y, Phi_mat, U, L);

hold on;
plot(x, Phi_mat * w_hat_spice) % just check the fit
%% 
%Find covariance parameters lambda0, lambda1, ..., lamda_d from SPICE theta
[lambda0, lambda_vec] = covariance_parameter_spice(y, Phi_mat, w_hat_spice);

%%
% Compute degrees of freedom for SPICE
df_spice = degrees_of_freedom(Phi_mat, lambda0, lambda_vec);

%%
% Compute degrees of freedon for LS
pseudo_inv = pinv(Phi_mat);
theta_ls = pseudo_inv * y;
df_ls = trace(Phi_mat * pseudo_inv);
hold on;
plot(x, Phi_mat * theta_ls);