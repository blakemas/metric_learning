n = 50;
p = 30;
d = 10;
num_samples = 4000;

% create some points
X = randn(n,p)/d^0.25;

% sparse K
K = zeros(p,p);
K(1:d,1:d) = eye(d);
G_s = X*K*X';

% dense K
U = orth(randn(p,p));
Sigma = zeros(p,p);
Sigma(1:d,1:d) = eye(d);
K = U*Sigma*U';
G_d = X*K*X';

sparse_difs = 0;
dense_difs = 0;
for t=1:num_samples
    q = datasample(1:n, 3, 'replace', false);
    i = q(1);
    j = q(2);
    k = q(3);
    sparse_difs = sparse_difs + abs(G_s(k,k) - 2*G_s(i,k) + 2*G_s(i,j) - G_s(j,j));
    dense_difs = dense_difs +  abs(G_d(k,k) - 2*G_d(i,k) + 2*G_d(i,j) - G_d(j,j));
end

% average triplet difference
sparse_difs/num_samples
dense_difs/num_samples