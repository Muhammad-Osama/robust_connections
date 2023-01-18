%%
% Define a grid of 2d points
nx1 = 25; nx2 = 25;
x1 = linspace(0, 10, nx1);
x2 = linspace(0, 10, nx2);
[X1, X2] = meshgrid(x1, x2);

% Define covarinace function
f3 = @(t) 1 + t;
sf = 2; len_sc = 7;  
r3 = @(x, z) sqrt( 3/(len_sc^2) * (x - z)' * (x - z) ); 
cov_matern_iso = @(x, z) sf^2 * f3 (r3(x, z)) * exp(-r3(x, z));

% Compute covariance matrix
X = [X1(:) X2(:)];
n = nx1 * nx2;
Kcov = zeros(n);
for i=1:n
    x = X(i,:)';
    for j = i:n
        z = X(j,:)';
        Kcov(j, i) = cov_matern_iso(x, z);
        Kcov(i, j) = Kcov(j, i);
    end
end
%%
seed = 10;
%%
% Sample a GP realization with above covariance matrix
se = 0.1;
rng(seed);
y = chol(Kcov)' * randn(n, 1) + se.* randn(n, 1);
y2d = reshape(y, nx1, nx1);

figure;
contourf(X1, X2, y2d, 10); colorbar
xlabel('$x_{1}$','interpreter','Latex');
ylabel('$x_{2}$','interpreter','Latex');

%%
rng(seed);
ntrain = 250;
xtrain = 10 .* rand(ntrain, 2);
hold on;
scatter(xtrain(:,1), xtrain(:, 2) , 50, 'rx')
%%
%Kcov_train = zeros(ntrain);
%for i=1:ntrain
%    x = xtrain(i,:)';
%    for j = i:ntrain
%        z = xtrain(j,:)';
%        Kcov_train(j, i) = cov_matern_iso(x, z);
%        Kcov_train(i, j) = Kcov_train(j, i);
%    end
%end
%rng(seed);
%ytrain = chol(Kcov_train)' * randn(ntrain, 1) + se.* randn(ntrain, 1);
%hold on;
%scatter(xtrain(:,1), xtrain(:, 2),[], ytrain, 'filled');
%%
% Sample training data points
rng(seed);
[ytrain, idx_train] = datasample(y, ntrain, 'Replace', false);
data_points_idx = 1:n;
bool_mask_test = ~ismember(data_points_idx, idx_train);
idx_test = data_points_idx(bool_mask_test);
Xtrain = X(idx_train, :);
ytest = y(idx_test);

%%
% Fit spice

%Define SPICE basis
basis_support = 0.75; L_vec = [10, 10];
nBasis = 10;
mn = [0 0]; mx = [10 10]; 
Phi_mat = zeros(n, 10^2);
for i = 1:n
    %Bspine basis
    %Phi_mat(i, :) = func_phi_bsplinebasis(X(i, :), mn, mx, nBasis, basis_support);
    %Laplace basis
    Phi_mat(i, :) = func_phi_laplacebasis(X(i,:), 10, L_vec);
end
Phi_train = Phi_mat(idx_train, :);
Phi_test = Phi_mat(idx_test, :);
%for i = 1:ntrain
%    Phi_train(i, :) = func_phi_bsplinebasis(xtrain(i, :), mn, mx, nBasis, basis_support);
%end
U = 1;
L = 10;
theta_spice = compute_spicepredictor(ytrain, Phi_train, U, L);

%% 
% predict all point just to see how SPICE prediction looks compared to the
% true function
y_hat_spice = Phi_mat * theta_spice;

figure;
contourf(X1, X2, reshape(y_hat_spice, 25, 25), 10); colorbar;
xlabel('$x_{1}$','interpreter','Latex');
ylabel('$x_{2}$','interpreter','Latex');

%%
% Compute error on test data for SPICE
y_spice_test = Phi_test * theta_spice;
error_spice = ytest - y_spice_test;
figure;
hist(error_spice); grid on;
xlabel('$y - \hat{y}$','interpreter','Latex');
mse_spice = mean(error_spice.^2);

%% 
% Use least-squares
theta_ls = pinv(Phi_train) * ytrain;
y_hat_ls = Phi_mat * theta_ls;
figure;
contourf(X1, X2, reshape(y_hat_ls, 25, 25), 10); colorbar;
xlabel('$x_{1}$','interpreter','Latex');
ylabel('$x_{2}$','interpreter','Latex');

%% 
% Compute error on test data for least-square
y_ls_test = Phi_test * theta_ls;
error_ls = ytest - y_ls_test;

figure;
hist(error_ls); grid on;
xlabel('$y - \hat{y}$','interpreter','Latex');
mse_ls = mean(error_ls.^2);

%%
%Use ridge regression
lambda = 0.1;
theta_rls = ((Phi_train'*Phi_train) + lambda .* eye(10^2))\(Phi_train' * ytrain);
y_hat_rls = Phi_mat * theta_rls;
figure;
contourf(X1, X2, reshape(y_hat_rls, 25, 25), 10); colorbar;
xlabel('$x_{1}$','interpreter','Latex');
ylabel('$x_{2}$','interpreter','Latex');

%%
% Compute error on test data for regularized least-square
y_rls_test = Phi_test * theta_rls;
error_rls = ytest - y_rls_test;

figure;
hist(error_rls); grid on;
xlabel('$y - \hat{y}$','interpreter','Latex');
mse_rls = mean(error_rls.^2);