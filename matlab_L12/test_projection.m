% make a matrix
n = 10;
% M = randn(n,n);
% B = M*M';

B = diag(abs(randn(n,1)));
L12(B)

lam = 20;

cvx_begin
    variable A(n,n) symmetric semidefinite;

    F = norm(A - B, 'fro');
    minimize(F);
    subject to
    L12_norm = 0;
    for i=1:n
        L12_norm = L12_norm + norm(A(i,:));
    end
    L12_norm <= lam;
cvx_end

% L12 norm of CVX learned matrix
L12(A)

norm(project_L12(B, lam) - A, 'fro')