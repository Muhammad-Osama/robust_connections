%%
seed = 0;
rng(seed);

%%
%Define SPICE basis parameters
basis_support = 0.75;
nBasis = 10;
mn = [0 0]; mx = [10 10];

% SPICE estimate parameters
U = 1; %mean
L = 10; %nos. of iterations

% Define covarinace function
f3 = @(t) 1 + t;
sf = 2; len_sc = 7;  
r3 = @(x, z) sqrt( 3/(len_sc^2) * (x - z)' * (x - z) ); 
cov_matern_iso = @(x, z) sf^2 * f3 (r3(x, z)) * exp(-r3(x, z));

%Uniformly sample train and test location
%m = 200; % nos. of test data points
n_range = 20:2:200; % nos. of training data points

df_spice = zeros(length(n_range), 1);
df_ls = zeros(length(n_range), 1);
df_gp = zeros(length(n_range), 1);
%%

for ni=1:length(n_range)
    
    n = n_range(ni) 
    
    %sample training locations
    xtrain = 10 * rand(n,2);  

    %sample test locations
    %xtest = 10 * rand(n, 2);

    % all sampling locations
    X = xtrain; 

    %
    % Compute covariance matrix
    Kcov = zeros(n);
    for i=1:n
        x = X(i,:)';
        for j = i:n
            z = X(j,:)';
            Kcov(j, i) = cov_matern_iso(x, z);
            Kcov(i, j) = Kcov(j, i);
        end
    end
    %
    % Sample a GP realization with above covariance matrix
    se = 0.1;
    rng(seed);
    ytrain = chol(Kcov)' * randn(n, 1) + se.* randn(n, 1);
    %ytrain = y(1:n); %ytest = y(n+1:end);
    %
    % Fit spice 
    Phi_train = zeros(n, 10^2);
    for i = 1: n
        Phi_train(i, :) = func_phi_bsplinebasis(X(i, :), mn, mx, nBasis, basis_support);
    end
    %Phi_train = Phi_mat(1:n, :);
    %Phi_test = Phi_mat(n+1:end, :);
    theta_spice = compute_spicepredictor(ytrain, Phi_train, U, L);
    %compute covariance parameters from SPICE
    [lambda0, lambda_vec] = covariance_parameter_spice(ytrain, Phi_train, theta_spice);
    %compute degrees of freedom
    df_spice(ni) = degrees_of_freedom(Phi_train, lambda0, lambda_vec);

    % Fit LS
    pseudo_inv = pinv(Phi_train);
    theta_ls = pseudo_inv * ytrain;
    df_ls(ni) = trace(Phi_train * pseudo_inv);
    
    % GP with true parameters
    df_gp(ni) = trace(Kcov * inv(Kcov + se^2.*eye(n)));
end
%%
figure;
plot(n_range, df_ls); hold on; 
plot(n_range, df_gp);
plot(n_range, df_spice); grid on;
legend({'LS', 'GP', 'SPICE'}, 'interpreter', 'Latex');
xlabel('$n$: nos. of data points','interpreter','Latex');
ylabel('$df$: degrees of freedom','interpreter','Latex')