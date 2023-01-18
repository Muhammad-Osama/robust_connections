seed = 0;
rng(seed);
%%
% Fixed parameters
n_range = 2:2:200; % Range of nos. of data points

%degrees of freedom
df_spice = zeros(length(n_range), 1);
df_ls = zeros(length(n_range), 1);
df_gp = zeros(length(n_range), 1);

se = 0.25; % std of error
basis_support = 1.5; 
nBasis = 100; % d
mn = 0; mx = 20; %range over which basis is defined

% SPICE predictor parameters
U = 1; % constant mean
L = 10; % nos. of itereations
%%
for i = 1: length(n_range)
    
    i
    
    % nos of data points
    n = n_range(i);
    % Generate data
    x = linspace(0, mx, n)';
    y = sin(2 * pi./mx .* x) + se .* randn(n, 1);
 
    %Fit SPICE
    %Define basis function parameters 
    Phi_mat = zeros(n, nBasis); % n x d matrix of basis
    for j = 1:n
        [Phi_mat(j, :), ~] = func_phi_bsplinebasis(x(j), mn, mx, nBasis, basis_support);
    end
    
    %Compute SPICE predictor
    w_hat_spice = compute_spicepredictor(y, Phi_mat, U, L);

    %Find covariance parameters lambda0, lambda1, ..., lamda_d from SPICE theta
    [lambda0, lambda_vec] = covariance_parameter_spice(y, Phi_mat, w_hat_spice);

    % Compute degrees of freedom for SPICE
    df_spice(i) = degrees_of_freedom(Phi_mat, lambda0, lambda_vec);
    
    % fit least squares
    pseudo_inv = pinv(Phi_mat);
    theta_ls = pseudo_inv * y;
    
    % degrees of freedom least-squares
    df_ls(i) = trace(Phi_mat * pseudo_inv);
    
    %Fit GP
    %meanfunc = [];
    %covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
    %likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
    
    %hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    
    %K = feval(covfunc{:}, hyp.cov, x);
    
    %sn = exp(hyp.lik);
    
    %df_gp(i) = trace(K * inv(K + sn^2 .* eye(n)));
    

end

%%
figure;
plot(n_range, df_ls); hold on; 
plot(n_range, df_spice); grid on;
%plot(n_range, df_gp);
legend({'LS', 'SPICE', 'GP'}, 'interpreter', 'Latex');
xlabel('$n$: nos. of data points','interpreter','Latex');
ylabel('$df$: degrees of freedom','interpreter','Latex')