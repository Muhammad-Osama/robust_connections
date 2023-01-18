%%
% In this m-file, we sample the training and test locations uniformly at
% random, then use them to compute a covariance matrix, then sample a GP
% using that covariance matrix, separate training and test data, fit SPICE
% and LS, compute mse using test data for each
%%
seed = 0;
%%
n = 500; % nos. of test points
m = 250; % nos. of test points

% Sampling locations
%rng(seed);
%xtrain = rand(n, 2);
%xtest = rand(m, 2);
%X = [xtrain;xtest];

%Define SPICE basis parameters
basis_support = 0.75;
nBasis = 10;
mn = [0 0]; mx = [10 10];
L_vec = [25, 25];

% Basis for fitting SPICE 
%Phi_mat = zeros(n + m, 10^2);
%for i = 1: n + m
%    Phi_mat(i, :) = func_phi_bsplinebasis(X(i, :), mn, mx, nBasis, basis_support);
%end
%Phi_train = Phi_mat(1:n, :);
%Phi_test = Phi_mat(n+1:end, :);

% SPICE estimate parameters
U = 1; %mean
L = 10; %nos. of iterations

% Define covarinace function
f3 = @(t) 1 + t;
sf = 2; len_sc = 7;  
r3 = @(x, z) sqrt( 3/(len_sc^2) * (x - z)' * (x - z) ); 
cov_matern_iso = @(x, z) sf^2 * f3 (r3(x, z)) * exp(-r3(x, z));

meanfunc = [];                    % empty: don't use a mean function
covfunc = {@covMaterniso, 3};          % Matern covariance function
likfunc = @likGauss; 
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

% Compute covariance matrix
%Kcov = zeros(n + m);
%for i=1:n + m
%   x = X(i,:)';
%   for j = i:n + m
%       z = X(j,:)';
%       Kcov(j, i) = cov_matern_iso(x, z);
%       Kcov(i, j) = Kcov(j, i);
%   end
%end

% Compute Oracle mse i.e. conditional variance at test points
se = 0.2;
%K_str_str = Kcov(n+1:end, n+1:end);
%K_str_x = Kcov(n+1:end, 1:n);
%K_x_str = Kcov(1:n, n+1:end);
%K_x_x = Kcov(1:n, 1:n);
%KI_inv_mat = inv((K_x_x + se^2.*eye(n)));
%var_str = diag(K_str_str) - diag(K_str_x * ((K_x_x + se.^2.*eye(n))\ K_x_str));
%mse_oracle = mean(var_str) + se.^2;

mc = 50; % nos. of Montecarlo runs
mse_test_spice = zeros(mc, 1); 
mse_test_ls = zeros(mc, 1);
mse_test_rls = zeros(mc, 1);
mse_oracle = zeros(mc, 1);
%mse_oracle = zeros(mc, 1);
finish_time = zeros(mc, 1);
%%
rng(seed);
for mci = 1 : mc
    mci
    
    %sample randomly
    xtrain = rand(n, 2);
    xtest = rand(m, 2);
    X = [xtrain;xtest];
   
    % Compute covariance matrix
    Kcov = zeros(n + m);
    for i=1:n + m
       x = X(i,:)';
       for j = i:n + m
           z = X(j,:)';
           Kcov(j, i) = cov_matern_iso(x, z);
           Kcov(i, j) = Kcov(j, i);
       end
    end

    % Sample GP
    y = chol(Kcov)' * randn(n + m, 1) + se.* randn(n + m, 1);
    ytrain = y(1:n); ytest = y(n+1:end);
   
    % Basis for fitting SPICE 
    
    Phi_mat = zeros(n + m, 10^2);
    for i = 1: n + m
        %Phi_mat(i, :) = func_phi_bsplinebasis(X(i, :), mn, mx, nBasis, basis_support);
        Phi_mat(i, :) = func_phi_laplacebasis(X(i,:), 10, L_vec);
    end
    Phi_train = Phi_mat(1:n, :);
    Phi_test = Phi_mat(n+1:end, :);
   
    % Fit spice
    
    theta_spice = compute_spicepredictor(ytrain, Phi_train, U, L);
    y_test_spice = Phi_test * theta_spice;

    mse_test_spice(mci) = mean((ytest - y_test_spice).^2);
    
    % Fit LS
    theta_ls = pinv(Phi_train) * ytrain;
    y_test_ls = Phi_test * theta_ls;
    mse_test_ls(mci) = mean((ytest - y_test_ls).^2);
    
    % Fit regularized LS (ridge regression);
    lambda = 0.1;
    tic;
    theta_rls = ((Phi_train'*Phi_train) + lambda .* eye(10^2))\(Phi_train' * ytrain);
    y_test_rls = Phi_test * theta_rls;
    finish_time(mci) = toc;
    mse_test_rls(mci) = mean((ytest - y_test_rls).^2);
    
    %compute oracle mse
    %hyp = minimize(hyp, @gp, -100, @infGaussLik, [], covfunc, likfunc, xtrain, ytrain);
    %[z1, z2] = meshgrid(linspace(0,1,10), linspace(0,1, 10));
    %z = [z1(:) z2(:)];
    %[mn s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, xtrain, ytrain, z);

    K_str_str = Kcov(n+1:end, n+1:end);
    K_str_x = Kcov(n+1:end, 1:n);
    K_x_str = Kcov(1:n, n+1:end);
    K_x_x = Kcov(1:n, 1:n);
    
    KI_inv_mat = inv((K_x_x + se^2.*eye(n)));
    var_str = diag(K_str_str) - diag(K_str_x * ((K_x_x + se.^2.*eye(n))\ K_x_str));
    mse_oracle(mci) = mean(var_str)+ se.^2;

    
end
%%
norm_mse_spice = mean(mse_test_spice./mse_oracle)
norm_mse_ls = mean(mse_test_ls./mse_oracle)
norm_mse_rls = mean(mse_test_rls./mse_oracle)
mean(finish_time)
%%
%figure;
%boxplot([mse_test_spice, mse_test_ls, mse_oracle], [1, 2, 3]); grid on; 
%xticklabels({'SPICE', 'LS', 'Oracle'})
%ylabel('Mean square error', 'interpreter','Latex');
