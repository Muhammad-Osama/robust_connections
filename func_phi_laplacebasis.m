function [Phi] = func_phi_laplacebasis( x, M, L_vec )
%% Phi(x) - using Laplace operator basis with rectangular boundaries
% x row vector R^d

%% Initialize
D   = size(x,2);
Phi = ones(1,M^D);

%% Construct Phi
j_vec = ones(1,D); %index

for c = 1:M^D
    
    %Product
    for k = 1:D
        Phi(c) = Phi(c) * sin( pi * j_vec(k) *  ( x(k) + L_vec(k) )/(2*L_vec(k)) )  / sqrt(L_vec(k));
    end
    
    
    %Index update
    j_vec(1) = j_vec(1) + 1;
    if D>1
        for k = 2:D
            if (mod(j_vec(k-1),M+1) == 0)
                j_vec(k-1) = 1;
                j_vec(k)   = j_vec(k) + 1;
            end
        end
    end

end

end

