n = 20;
p = 4;
d = 2;
num_triplets = 2500;
ranks = [];
errs = [];
% norm_L12 = @(K) sqrt(sum(abs(K).^2,2));

X = randn(p,n)/d^0.25;
Ktrue = zeros(p,p);
Ktrue(1:d,1:d) = eye(d);
S = build_triples(Ktrue, X, num_triplets);

nuc_Khat = triplets(Ktrue, X, S, 'nuclear');
nuc_err = norm(Ktrue-nuc_Khat,'fro')^2/norm(Ktrue,'fro')^2
nuc_excess_risk = excess_risk(Ktrue, nuc_Khat, X)

L12_Khat = triplets(Ktrue, X, S, 'L12');
L12_err = norm(Ktrue-L12_Khat,'fro')^2/norm(Ktrue,'fro')^2
L12_excess_risk = excess_risk(Ktrue, L12_Khat, X)


