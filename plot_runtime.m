n = [50, 100, 250, 500];

runtime_gp = [59, 113.8, 231.3, 1507.2];

runtime_ridge = [0.15, 0.16, 0.20, 0.22];

runtime_spice = [6, 11.6, 29.2, 58.6];

loglog(n, runtime_gp, 'LineWidth', 1.5);
hold on;
loglog(n, runtime_ridge, 'LineWidth', 1.5)
loglog(n, runtime_spice, 'LineWidth', 1.5)
xlabel('$n:$ nos. of data points', 'interpreter', 'Latex')
ylabel('Run time $(ms)$', 'interpreter', 'Latex')
legend('GP', 'Ridge', 'Spice')
grid on;