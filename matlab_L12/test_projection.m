load data.mat
B = sparse(double(Mtrue));

% B = diag(abs(randn(n,1)));
L12(B);

cvx_begin
%     lam = 5;
    variable A(n,n);
    
    variable C(n,n);
    for i=1:n
        for j=1:n
            C(i,j) == A(i,j) - B(i,j);
        end
    end
    
    minimize(norm(C, 'fro'));
    subject to
    L12_norm = 0;
    for i=1:n
        L12_norm = L12_norm + norm(A(i,:));
    end
    L12_norm <= lam;
cvx_end

% L12 norm of CVX learned matrix
L12(A)
L12(Mproj)
norm(Mproj - A, 'fro')

figure; 
imagesc(A)
figure;
imagesc(Mproj)
