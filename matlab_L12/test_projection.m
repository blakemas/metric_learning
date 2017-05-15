% make a matrix
% n = 10;
% M = randn(n,n);
% B = M*M';
load data.mat
B = zeros(n,n);
for i=1:n
    for j=1:n
        B(i,j) = Mtrue(i,j);
    end
end

% B = diag(abs(randn(n,1)));
%L12(B);

cvx_begin
    variable A(10,10) semidefinite symmetric; 
    minimize(norm(A-B, 'fro'));
    subject to
    L12_norm = 0;
    for i=1:n
        L12_norm = L12_norm + norm(A(i,:));
    end
    L12_norm <= 5;
cvx_end

% L12 norm of CVX learned matrix
L12(A)
L12(Mproj)
norm(Mproj - A, 'fro')/norm(Mproj,'fro')

figure; 
subplot(121)
imagesc(A)
subplot(122)
imagesc(Mproj)